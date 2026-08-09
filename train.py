import os
import time
import random
import argparse
import yaml
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import network_parameters, losses
import utils.losses
from warmup_scheduler import GradualWarmupScheduler
from transform.data_RGB import get_training_data, get_validation_data2

# === MODEL & LOSSES ===
from model.NightFormer import NightFormer
import Myloss

import os
import torch

import os
import torch

import os
import torch

import os
import torch

import os
import torch

import os
import torch

import os
import torch

import os
import torch

# CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()  # Clear GPU cache
        return device
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available, using CPU")
        return device

device = setup_device()


# CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()  # Clear GPU cache
        return device
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available, using CPU")
        return device

device = setup_device()


# CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()  # Clear GPU cache
        return device
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available, using CPU")
        return device

device = setup_device()


# CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()  # Clear GPU cache
        return device
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available, using CPU")
        return device

device = setup_device()


# CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()  # Clear GPU cache
        return device
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available, using CPU")
        return device

device = setup_device()


# CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()  # Clear GPU cache
        return device
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available, using CPU")
        return device

device = setup_device()


# CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()  # Clear GPU cache
        return device
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available, using CPU")
        return device

device = setup_device()


# CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()  # Clear GPU cache
        return device
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available, using CPU")
        return device

device = setup_device()


# ---------------------------
# Device / GPU setup
# ---------------------------
def setup_device(gpu_list_from_yaml):
    # Respect YAML GPU list (e.g., [0] or [0,1])
    gpus = ','.join([str(i) for i in gpu_list_from_yaml]) if gpu_list_from_yaml else '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("⚠ CUDA not available, using CPU")
    return device

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hyper-parameters for NightFormer')
    parser.add_argument('-yml_path', default="./configs/NightFormer.yaml", type=str)
    args = parser.parse_args()

    # Determinism-ish
    torch.backends.cudnn.benchmark = True
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    # Load YAML
    yaml_file = args.yml_path
    with open(yaml_file, 'r') as config:
        opt = yaml.safe_load(config)
    print("load training yaml file: %s" % yaml_file)

    Train = opt['TRAINING']
    OPT   = opt['OPTIM']
    MODE  = opt['MODEL']['MODE']

    # Device (respect YAML GPU list)
    device = setup_device(opt.get('GPU', [0]))
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"\n\nLet's use {torch.cuda.device_count()} GPUs!\n\n")

    # ---------------------------
    # Build model (use new API)
    # ---------------------------
    print('==> Build the model')
    base_ch = opt['MODEL'].get('BASE_CH', 48)
    use_weather_adapters = bool(opt['MODEL'].get('USE_WEATHER_ADAPTERS', True))

    model_NightFormer = NightFormer(base_ch=base_ch, use_weather_adapters=use_weather_adapters)
    para_number = network_parameters(model_NightFormer)

    # DataParallel (optional)
    if len(device_ids) > 1:
        model_NightFormer = nn.DataParallel(model_NightFormer, device_ids=device_ids)
    model_NightFormer = model_NightFormer.to(device)

    # ---------------------------
    # Paths
    # ---------------------------
    mode = MODE
    model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
    os.makedirs(model_dir, exist_ok=True)

    train_dir = Train['TRAIN_DIR']
    val_dir   = Train['VAL_DIR']

    # ---------------------------
    # Optimizer & Scheduler
    # ---------------------------
    start_epoch = 1
    new_lr = float(OPT['LR_INITIAL'])
    optimizer = optim.Adam(model_NightFormer.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, OPT['EPOCHS'] - warmup_epochs, eta_min=float(OPT['LR_MIN'])
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine
    )
    # IMPORTANT: do NOT call scheduler.step() here before first optimizer.step()
    # We'll step the scheduler at the END of each epoch.

    # ---------------------------
    # Resume (optional)
    # ---------------------------
    if Train['RESUME']:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_NightFormer, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        # Advance scheduler to current epoch (safe to do now)
        for _ in range(1, start_epoch):
            optimizer.step()
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------')

    # ---------------------------
    # Losses
    # ---------------------------
    L_SL1 = nn.SmoothL1Loss()
    L_col = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16)
    L_ssim = Myloss.SSIM()
    L_per  = Myloss.VGGLoss(device)

    # ---------------------------
    # Data
    # ---------------------------
    print('==> Loading datasets')
    train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
    train_loader  = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                               shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_dataset   = get_validation_data2(val_dir, {'patch_size': Train['VAL_PS']})
    val_loader    = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                               num_workers=0, drop_last=False)

    # ---------------------------
    # Show config
    # ---------------------------
    print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {para_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}
------------------------------------------------------------------''')

    # ---------------------------
    # Logging
    # ---------------------------
    print('==> Training start: ')
    best_PSNR = 0
    best_SSIM = 0
    best_epoch_PSNR = 0
    best_epoch_SSIM = 0
    total_start_time = time.time()

    log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
        epoch_start_time = time.time()
        epoch_loss = 0.0

        model_NightFormer.train()

        for i, data in enumerate(tqdm(train_loader), 0):
            optimizer.zero_grad(set_to_none=True)

            # Your dataloader returns (target, input) → keep as is
            target = data[0].to(device)      # GT
            input_img = data[1].to(device)   # degraded

            # --- forward (UNPACK TUPLE) ---
            enhanced_img, aux = model_NightFormer(input_img)  # enhanced_img: (B,3,H,W)

            # --- losses ---
            E = 0.6  # exposure target
            loss_SL1  = L_SL1(enhanced_img, target)
            loss_ssim = 1 - L_ssim(enhanced_img, target)
            loss_spa  = torch.mean(L_spa(input_img, enhanced_img))
            loss_col  = 5 * torch.mean(L_col(enhanced_img))
            loss_exp  = 10 * torch.mean(L_exp(enhanced_img, E))
            loss_per  = L_per(enhanced_img, target)

            # You can weight as you prefer; this follows your previous pattern
            loss = loss_SL1 + 0.1 * loss_ssim + 0.1 * loss_spa + 0.1 * loss_per
            # (Optionally include loss_col, loss_exp with small weights if helpful.)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # ---------------------------
        # Validation
        # ---------------------------
        if epoch % Train['VAL_AFTER_EVERY'] == 0:
            model_NightFormer.eval()
            PSNR_val_rgb = []
            SSIM_val_rgb = []
            with torch.no_grad():
                for ii, data_val in enumerate(val_loader, 0):
                    target = data_val[0].to(device)
                    input_img = data_val[1].to(device)
                    h, w = target.shape[2], target.shape[3]

                    pred_img, aux = model_NightFormer(input_img)     # UNPACK
                    pred_img = pred_img[:, :, :h, :w]               # crop if needed

                    # compute metrics per-sample to avoid broadcasting mistakes
                    for res, tar in zip(pred_img, target):
                        PSNR_val_rgb.append(utils.torchPSNR(res, tar))
                        SSIM_val_rgb.append(utils.torchSSIM(res.unsqueeze(0), tar.unsqueeze(0)))

            PSNR_val_rgb = torch.stack(PSNR_val_rgb).mean().item() if PSNR_val_rgb else 0.0
            SSIM_val_rgb = torch.stack(SSIM_val_rgb).mean().item() if SSIM_val_rgb else 0.0

            # Save best PSNR
            if PSNR_val_rgb > best_PSNR:
                best_PSNR = PSNR_val_rgb
                best_epoch_PSNR = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_NightFormer.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestPSNR.pth"))
            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                  (epoch, PSNR_val_rgb, best_epoch_PSNR, best_PSNR))

            # Save best SSIM
            if SSIM_val_rgb > best_SSIM:
                best_SSIM = SSIM_val_rgb
                best_epoch_SSIM = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_NightFormer.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestSSIM.pth"))
            print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" %
                  (epoch, SSIM_val_rgb, best_epoch_SSIM, best_SSIM))

            writer.add_scalar('val/PSNR', PSNR_val_rgb, epoch)
            writer.add_scalar('val/SSIM', SSIM_val_rgb, epoch)

        # Scheduler step AFTER optimizer stepped this epoch
        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
            epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        # Save latest
        torch.save({'epoch': epoch,
                    'state_dict': model_NightFormer.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))

        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

    writer.close()

    total_finish_time = (time.time() - total_start_time)
    print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))


