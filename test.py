import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import numpy as np
import re

import cv2
import argparse
from model.NightFormer import NightFormer

parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--input_dir', default='./datasets/test/input/', type=str, help='Input images')
parser.add_argument('--target_dir', default='./datasets/test/target/', type=str, help='Target/GT images')
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

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return cv2.PSNR(img1, img2)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, channel_axis=2, data_range=255)


def detect_modality(filename):
    """Detect modality based on filename pattern at the END of the filename"""
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    
    # Match pattern at the end: _0, _00, _000, _0000, etc.
    # Use word boundary to ensure we match the ending pattern
    match = re.search(r'_(0+)$', base_name)
    
    if match:
        zeros = match.group(1)
        num_zeros = len(zeros)
        
        # Map number of zeros to modality
        modality_map = {
            1: 'Fog',
            2: 'Haze',
            3: 'Rain',
            4: 'Snow'
        }
        
        return modality_map.get(num_zeros, f'Unknown_{num_zeros}zeros')
    
    return 'Unknown'

inp_dir = args.input_dir
target_dir = args.target_dir
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

# Get target files
target_files = natsorted(glob(os.path.join(target_dir, '*.jpg')) +
                        glob(os.path.join(target_dir, '*.JPG')) +
                        glob(os.path.join(target_dir, '*.png')) +
                        glob(os.path.join(target_dir, '*.PNG')) +
                        glob(os.path.join(target_dir, '*.bmp')))

# Check if target directory exists
if not os.path.exists(target_dir):
    print(f"\n❌ ERROR: Target directory does not exist: {target_dir}")
    print("Please provide the correct path to ground truth images using --target_dir")
    exit(1)

print('=' * 80)
print('DATASET INFORMATION')
print('=' * 80)
print(f"Input directory: {inp_dir}")
print(f"Target directory: {target_dir}")
print(f"Output directory: {out_dir}")
print(f"\nNumber of input files: {len(files)}")
print(f"Number of target files: {len(target_files)}")

# Detect modalities from filenames
modality_preview = {}
for f in files[:10]:  # Preview first 10
    mod = detect_modality(os.path.basename(f))
    modality_preview[mod] = modality_preview.get(mod, 0) + 1

print(f"\nDetected modalities (from first 10 files):")
for mod, count in modality_preview.items():
    print(f"  {mod}: {count} files")
print('=' * 80)

# Load model
model = NightFormer(base_ch=48, use_weather_adapters=True)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

# Initialize metrics storage
modality_metrics = {}
overall_psnr = []
overall_ssim = []

print('\nRestoring images and calculating metrics......')
print('=' * 80)

mul = 16
index = 0

for file_ in files:
    # Load input image
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    # Pad the input if not_multiple_of 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    
    # Inference
    with torch.no_grad():
        restored, _ = model(input_)  # Unpack tuple

    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    # Save restored image
    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.jpg')), restored)

    # Find corresponding target image
    base_filename = os.path.basename(file_)
    target_file = None
    
    # Try exact match first
    potential_target = os.path.join(target_dir, base_filename)
    if os.path.exists(potential_target):
        target_file = potential_target
    else:
        # Try different extensions
        base_name = os.path.splitext(base_filename)[0]
        for ext in ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP']:
            potential_target = os.path.join(target_dir, base_name + ext)
            if os.path.exists(potential_target):
                target_file = potential_target
                break
    
    if target_file and os.path.exists(target_file):
        target_img = cv2.imread(target_file)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        # Resize target if dimensions don't match
        if target_img.shape[:2] != restored.shape[:2]:
            target_img = cv2.resize(target_img, (restored.shape[1], restored.shape[0]))
        
        # Calculate PSNR and SSIM
        psnr_value = calculate_psnr(target_img, restored)
        ssim_value = calculate_ssim(target_img, restored)
        
        # Add to overall metrics
        overall_psnr.append(psnr_value)
        overall_ssim.append(ssim_value)
        
        # Detect modality and store metrics
        modality = detect_modality(base_filename)
        if modality not in modality_metrics:
            modality_metrics[modality] = {'psnr': [], 'ssim': []}
        
        modality_metrics[modality]['psnr'].append(psnr_value)
        modality_metrics[modality]['ssim'].append(ssim_value)
        
        status = f"✓ [{modality}] PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}"
    else:
        status = "✗ No target found"
    
    index += 1
    print(f'{index}/{len(files)} - {f} {status}')

print('=' * 80)
print(f"\nFiles saved at {out_dir}")

# Print metrics for each modality
if modality_metrics:
    print('\n' + '=' * 80)
    print('METRICS PER MODALITY')
    print('=' * 80)

    for modality in sorted(modality_metrics.keys()):
        metrics = modality_metrics[modality]
        num_images = len(metrics['psnr'])
        avg_psnr = np.mean(metrics['psnr'])
        avg_ssim = np.mean(metrics['ssim'])
        
        print(f"\n{modality.upper()} ({num_images} images)")
        print(f"  Average PSNR: {avg_psnr:.4f} dB")
        print(f"  Average SSIM: {avg_ssim * 100:.4f} %")

# Print overall average
print('\n' + '=' * 80)
print('OVERALL METRICS')
print('=' * 80)
if len(overall_psnr) > 0:
    print(f"\nTotal images processed: {len(overall_psnr)}")
    print(f"Overall Average PSNR: {np.mean(overall_psnr):.4f} dB")
    print(f"Overall Average SSIM: {np.mean(overall_ssim) * 100:.4f} %")
else:
    print("\n⚠️ No target images found for evaluation")

print('\n' + '=' * 80)
print('Finish!')
print('=' * 80)


