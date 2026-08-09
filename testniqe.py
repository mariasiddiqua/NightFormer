import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import numpy as np
import re

import cv2
import argparse
from model.NightFormer import NightFormer

# pip install pyiqa
import pyiqa

parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--input_dir', default='./datasets/test/input/', type=str, help='Input images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/NightFormer/models/model_bestPSNR.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def detect_modality(filename):
    """Detect modality based on filename pattern at the END of the filename"""
    base_name = os.path.splitext(filename)[0]
    match = re.search(r'_(0+)$', base_name)

    if match:
        zeros = match.group(1)
        num_zeros = len(zeros)
        modality_map = {
            1: 'Fog',
            2: 'Haze',
            3: 'Rain',
            4: 'Snow'
        }
        return modality_map.get(num_zeros, f'Unknown_{num_zeros}zeros')

    return 'Unknown'

# ─── Setup ────────────────────────────────────────────────────────────────────
inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

# Get input files
files = natsorted(glob(os.path.join(inp_dir, '*.jpg')) +
                  glob(os.path.join(inp_dir, '*.JPG')) +
                  glob(os.path.join(inp_dir, '*.png')) +
                  glob(os.path.join(inp_dir, '*.PNG')) +
                  glob(os.path.join(inp_dir, '*.bmp')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

print('=' * 80)
print('DATASET INFORMATION')
print('=' * 80)
print(f"Input directory:  {inp_dir}")
print(f"Output directory: {out_dir}")
print(f"Number of input files: {len(files)}")

# Detect modalities from filenames
modality_preview = {}
for f in files[:10]:
    mod = detect_modality(os.path.basename(f))
    modality_preview[mod] = modality_preview.get(mod, 0) + 1

print(f"\nDetected modalities (from first 10 files):")
for mod, count in modality_preview.items():
    print(f"  {mod}: {count} files")
print('=' * 80)

# ─── Load Model ───────────────────────────────────────────────────────────────
model = NightFormer(base_ch=48, use_weather_adapters=True)
model.cuda()
load_checkpoint(model, args.weights)
model.eval()

# ─── Load NIQE & PIQE metrics (runs on CPU, expects [B, C, H, W] in [0,1]) ───
niqe_metric = pyiqa.create_metric('niqe', device='cpu')  # lower is better
piqe_metric = pyiqa.create_metric('piqe', device='cpu')  # lower is better

# ─── Inference loop ───────────────────────────────────────────────────────────
modality_metrics = {}
overall_niqe = []
overall_piqe = []

print('\nRestoring images and calculating metrics......')
print('=' * 80)

mul = 16

for index, file_ in enumerate(files, 1):
    # ── Load & pad input ──────────────────────────────────────────────────────
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    h, w = input_.shape[2], input_.shape[3]
    H = ((h + mul - 1) // mul) * mul
    W = ((w + mul - 1) // mul) * mul
    padh = H - h
    padw = W - w
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        restored, _ = model(input_)

    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]  # remove padding

    # ── Save restored image (uint8 version) ───────────────────────────────────
    restored_np = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored_uint8 = img_as_ubyte(restored_np[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img(os.path.join(out_dir, f + '.jpg'), restored_uint8)

    # ── Calculate NIQE & PIQE on the float tensor (no target needed) ──────────
    # pyiqa expects: [B, C, H, W] float tensor in [0, 1], on the metric's device
    restored_cpu = restored.cpu()

    niqe_value = niqe_metric(restored_cpu).item()
    piqe_value = piqe_metric(restored_cpu).item()

    overall_niqe.append(niqe_value)
    overall_piqe.append(piqe_value)

    # ── Store per-modality ────────────────────────────────────────────────────
    modality = detect_modality(os.path.basename(file_))
    if modality not in modality_metrics:
        modality_metrics[modality] = {'niqe': [], 'piqe': []}

    modality_metrics[modality]['niqe'].append(niqe_value)
    modality_metrics[modality]['piqe'].append(piqe_value)

    print(f'{index}/{len(files)} - {f}  |  [{modality}]  NIQE: {niqe_value:.4f}  PIQE: {piqe_value:.4f}')

print('=' * 80)
print(f"\nRestored images saved at: {out_dir}")

# ─── Per-modality summary ─────────────────────────────────────────────────────
if modality_metrics:
    print('\n' + '=' * 80)
    print('METRICS PER MODALITY  (lower NIQE & PIQE = better quality)')
    print('=' * 80)

    for modality in sorted(modality_metrics.keys()):
        metrics = modality_metrics[modality]
        num_images = len(metrics['niqe'])
        avg_niqe = np.mean(metrics['niqe'])
        avg_piqe = np.mean(metrics['piqe'])

        print(f"\n  {modality.upper()} ({num_images} images)")
        print(f"    Avg NIQE: {avg_niqe:.4f}")
        print(f"    Avg PIQE: {avg_piqe:.4f}")

# ─── Overall summary ──────────────────────────────────────────────────────────
print('\n' + '=' * 80)
print('OVERALL METRICS  (lower NIQE & PIQE = better quality)')
print('=' * 80)
if len(overall_niqe) > 0:
    print(f"\n  Total images processed : {len(overall_niqe)}")
    print(f"  Overall Avg NIQE       : {np.mean(overall_niqe):.4f}")
    print(f"  Overall Avg PIQE       : {np.mean(overall_piqe):.4f}")
else:
    print("\n⚠️  No images were processed.")

print('\n' + '=' * 80)
print('Finish!')
print('=' * 80)

