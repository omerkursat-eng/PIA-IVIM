"""
Hızlı test - k-space'den parametre tahmini
"""

import numpy as np
import torch
from scipy import fft
from PIA import PIA
from utils import hybrid_fit

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# b-değerleri
b_values = [0, 5, 50, 100, 200, 500, 800, 1000]

# Veri yükleme
print("\n1. Veri yükleme...")
ivim_params = np.load('data/0001_IVIMParam.npy')
noisy_kspace = np.load('data/0001_NoisyDWIk.npy')
tissue_type = np.load('data/0001_TissueType.npy')

# Ground truth (1000 ile çarpılmış)
f_gt = ivim_params[:, :, 0]
Dt_gt = ivim_params[:, :, 1] * 1000
Dstar_gt = ivim_params[:, :, 2] * 1000

print(f"Original Dt range: [{ivim_params[:, :, 1].min():.6f}, {ivim_params[:, :, 1].max():.6f}]")
print(f"Scaled Dt range: [{Dt_gt.min():.4f}, {Dt_gt.max():.4f}]")
print(f"Original Dstar range: [{ivim_params[:, :, 2].min():.6f}, {ivim_params[:, :, 2].max():.6f}]")
print(f"Scaled Dstar range: [{Dstar_gt.min():.4f}, {Dstar_gt.max():.4f}]")

# k-space'den rekonstrüksiyon
print("\n2. k-space rekonstrüksiyonu...")
image_data = fft.ifft2(noisy_kspace, axes=(0, 1))
# FFT scaling düzeltmesi
H, W = noisy_kspace.shape[:2]
image_data = image_data * (H * W)
magnitude_signal = np.abs(image_data)
print(f"Magnitude range (raw): [{magnitude_signal.min():.6f}, {magnitude_signal.max():.6f}]")

# Normalize by b=0
magnitude_signal_normalized = magnitude_signal / (magnitude_signal[:, :, [0]] + 1e-8)
print(f"Magnitude range (normalized): [{magnitude_signal_normalized.min():.6f}, {magnitude_signal_normalized.max():.6f}]")

# Test voxel seç (tissue olan bir yer)
print("\n3. Test voxel seçimi...")
test_coords = []
for i in range(100, 105):
    for j in range(100, 105):
        if tissue_type[i, j] > 0:
            test_coords.append((i, j))
            if len(test_coords) >= 5:
                break
    if len(test_coords) >= 5:
        break

print(f"Selected {len(test_coords)} test voxels")

# NLLS test
print("\n4. NLLS (Hybrid Fit) test...")
for idx, (i, j) in enumerate(test_coords):
    signal = magnitude_signal_normalized[i, j, :].reshape(1, -1)
    f_nlls, Dt_nlls, Dstar_nlls = hybrid_fit(signal, bvals=b_values)
    
    print(f"\nVoxel [{i}, {j}] - Tissue type: {tissue_type[i, j]}")
    print(f"  GT:   f={f_gt[i,j]:.4f}, Dt={Dt_gt[i,j]:.4f}, D*={Dstar_gt[i,j]:.4f}")
    print(f"  NLLS: f={f_nlls[0]:.4f}, Dt={Dt_nlls[0]:.4f}, D*={Dstar_nlls[0]:.4f}")
    print(f"  Error: Δf={abs(f_nlls[0]-f_gt[i,j]):.4f}, ΔDt={abs(Dt_nlls[0]-Dt_gt[i,j]):.4f}, ΔD*={abs(Dstar_nlls[0]-Dstar_gt[i,j]):.4f}")

# PIA model test
print("\n5. PIA Model test...")
model_path = 'pia_runs/exp2_balanced_weighted/best_high_noise.pt'
model = PIA(
    number_of_signals=8,
    f_mean=0.5, f_delta=0.5,
    Dt_mean=1.45, Dt_delta=1.45,
    Dstar_mean=30, Dstar_delta=30,
    b_values=b_values,
    predictor_depth=6,
    device=device
).to(device)

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

print(f"Model loaded from: {model_path}")

with torch.no_grad():
    for idx, (i, j) in enumerate(test_coords):
        signal = magnitude_signal_normalized[i, j, :]
        signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(device)
        
        f_pia, Dt_pia, Dstar_pia = model.encode(signal_tensor)
        f_pia = f_pia.cpu().numpy()[0, 0]
        Dt_pia = Dt_pia.cpu().numpy()[0, 0]
        Dstar_pia = Dstar_pia.cpu().numpy()[0, 0]
        
        print(f"\nVoxel [{i}, {j}]")
        print(f"  GT:  f={f_gt[i,j]:.4f}, Dt={Dt_gt[i,j]:.4f}, D*={Dstar_gt[i,j]:.4f}")
        print(f"  PIA: f={f_pia:.4f}, Dt={Dt_pia:.4f}, D*={Dstar_pia:.4f}")
        print(f"  Error: Δf={abs(f_pia-f_gt[i,j]):.4f}, ΔDt={abs(Dt_pia-Dt_gt[i,j]):.4f}, ΔD*={abs(Dstar_pia-Dstar_gt[i,j]):.4f}")

print("\n✓ Test tamamlandı!")
