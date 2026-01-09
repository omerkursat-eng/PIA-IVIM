"""
k-space rekonstrüksiyon yöntemlerini test et
"""

import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

print("k-space Rekonstrüksiyon Test")
print("="*60)

# Veri yükle
noisy_kspace = np.load('data/0001_NoisyDWIk.npy')
gt_dwis = np.load('data/0001_gtDWIs.npy')

i, j = 100, 100  # Test voxel

print(f"\n1. k-space verisi:")
print(f"   Shape: {noisy_kspace.shape}")
print(f"   Dtype: {noisy_kspace.dtype}")
print(f"   Is complex: {np.iscomplexobj(noisy_kspace)}")

# Farklı rekonstrüksiyon yöntemleri
print(f"\n2. Rekonstrüksiyon yöntemleri:")

# Yöntem 1: ifft2 + ifftshift
method1 = fft.ifft2(noisy_kspace, axes=(0, 1))
method1 = fft.ifftshift(method1, axes=(0, 1))
mag1 = np.abs(method1)
print(f"\n   Method 1 (ifft2 + ifftshift):")
print(f"   Magnitude [{i},{j}]: {mag1[i, j, :]}")
print(f"   Range: [{mag1.min():.6f}, {mag1.max():.6f}]")

# Yöntem 2: fftshift önce, sonra ifft2
method2 = fft.fftshift(noisy_kspace, axes=(0, 1))
method2 = fft.ifft2(method2, axes=(0, 1))
mag2 = np.abs(method2)
print(f"\n   Method 2 (fftshift + ifft2):")
print(f"   Magnitude [{i},{j}]: {mag2[i, j, :]}")
print(f"   Range: [{mag2.min():.6f}, {mag2.max():.6f}]")

# Yöntem 3: Sadece ifft2
method3 = fft.ifft2(noisy_kspace, axes=(0, 1))
mag3 = np.abs(method3)
print(f"\n   Method 3 (sadece ifft2):")
print(f"   Magnitude [{i},{j}]: {mag3[i, j, :]}")
print(f"   Range: [{mag3.min():.6f}, {mag3.max():.6f}]")

# Yöntem 4: ifftshift önce, sonra ifft2
method4 = fft.ifftshift(noisy_kspace, axes=(0, 1))
method4 = fft.ifft2(method4, axes=(0, 1))
mag4 = np.abs(method4)
print(f"\n   Method 4 (ifftshift + ifft2):")
print(f"   Magnitude [{i},{j}]: {mag4[i, j, :]}")
print(f"   Range: [{mag4.min():.6f}, {mag4.max():.6f}]")

# Ground truth
mag_gt = np.abs(gt_dwis)
print(f"\n   Ground Truth:")
print(f"   Magnitude [{i},{j}]: {mag_gt[i, j, :]}")
print(f"   Range: [{mag_gt.min():.6f}, {mag_gt.max():.6f}]")

# En iyi yöntemi bul (GT'ye en yakın)
methods = [mag1, mag2, mag3, mag4]
method_names = ['ifft2+ifftshift', 'fftshift+ifft2', 'ifft2', 'ifftshift+ifft2']

print(f"\n3. Hangi yöntem GT'ye daha yakın?")
for idx, (mag, name) in enumerate(zip(methods, method_names)):
    # Normalize et
    mag_norm = mag / (mag[:, :, [0]] + 1e-8)
    gt_norm = mag_gt / (mag_gt[:, :, [0]] + 1e-8)
    
    # RMSE hesapla
    rmse = np.sqrt(np.mean((mag_norm - gt_norm) ** 2))
    
    # Test voxel için korelasyon
    corr = np.corrcoef(mag_norm[i, j, :], gt_norm[i, j, :])[0, 1]
    
    print(f"   {name:20s}: RMSE={rmse:.6f}, Corr[{i},{j}]={corr:.4f}")

# Görselleştirme
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (mag, name) in enumerate(zip(methods, method_names)):
    row = idx // 3
    col = idx % 3
    
    # b=0 görüntüsü
    im = axes[row, col].imshow(mag[:, :, 0], cmap='gray')
    axes[row, col].set_title(f'{name}\nb=0 image')
    axes[row, col].axis('off')
    plt.colorbar(im, ax=axes[row, col])

# GT
im = axes[1, 2].imshow(mag_gt[:, :, 0], cmap='gray')
axes[1, 2].set_title('Ground Truth\nb=0 image')
axes[1, 2].axis('off')
plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('kspace_reconstruction_comparison.png', dpi=150)
print(f"\n4. Görsel kaydedildi: kspace_reconstruction_comparison.png")

# Sinyal decay eğrileri
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
b_values = [0, 5, 50, 100, 200, 500, 800, 1000]

for idx, (mag, name) in enumerate(zip(methods, method_names)):
    row = idx // 2
    col = idx % 2
    
    # Normalize
    mag_norm = mag[i, j, :] / (mag[i, j, 0] + 1e-8)
    gt_norm = mag_gt[i, j, :] / mag_gt[i, j, 0]
    
    axes[row, col].plot(b_values, gt_norm, 'o-', label='GT', linewidth=2, markersize=8)
    axes[row, col].plot(b_values, mag_norm, 's-', label=name, linewidth=2, markersize=6)
    axes[row, col].set_xlabel('b-value (s/mm²)')
    axes[row, col].set_ylabel('Normalized Signal')
    axes[row, col].set_title(f'{name} - Voxel [{i},{j}]')
    axes[row, col].legend()
    axes[row, col].grid(True)
    axes[row, col].set_ylim([-0.5, 6])

plt.tight_layout()
plt.savefig('signal_decay_comparison.png', dpi=150)
print(f"5. Görsel kaydedildi: signal_decay_comparison.png")

print("\n" + "="*60)
print("Test tamamlandı!")
