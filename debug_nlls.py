"""
NLLS fitting problemini debug etmek için
"""

import numpy as np
from scipy import fft
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# IVIM model (utils.py'deki gibi)
def ivim_fit_func(b, f, Dt, Dstar):
    """IVIM bi-exponential model for curve fitting."""
    return (1 - f) * np.exp(-b/1000 * Dt) + f * np.exp(-b/1000 * Dstar)

# b-değerleri
b_values = np.array([0, 5, 50, 100, 200, 500, 800, 1000], dtype=np.float32)

# Veri yükleme
print("1. Veri yükleme...")
ivim_params = np.load('data/0001_IVIMParam.npy')
noisy_kspace = np.load('data/0001_NoisyDWIk.npy')
gt_dwis = np.load('data/0001_gtDWIs.npy')

# Ground truth (1000 ile çarpılmış)
f_gt = ivim_params[:, :, 0]
Dt_gt = ivim_params[:, :, 1] * 1000
Dstar_gt = ivim_params[:, :, 2] * 1000

# Test voxel
i, j = 100, 100
print(f"\nTest Voxel [{i}, {j}]:")
print(f"  GT: f={f_gt[i,j]:.4f}, Dt={Dt_gt[i,j]:.4f}, D*={Dstar_gt[i,j]:.4f}")

# k-space'den rekonstrüksiyon
print("\n2. k-space rekonstrüksiyonu...")
image_data = fft.ifft2(noisy_kspace, axes=(0, 1))
image_data = fft.ifftshift(image_data, axes=(0, 1))
magnitude_noisy = np.abs(image_data)

# Ground truth magnitude
magnitude_gt = np.abs(gt_dwis)

print(f"\nGround truth sinyal [{i},{j}]:")
print(f"  Values: {magnitude_gt[i, j, :]}")
print(f"  Range: [{magnitude_gt[i, j, :].min():.6f}, {magnitude_gt[i, j, :].max():.6f}]")

print(f"\nGürültülü sinyal (k-space'den) [{i},{j}]:")
print(f"  Values: {magnitude_noisy[i, j, :]}")
print(f"  Range: [{magnitude_noisy[i, j, :].min():.6f}, {magnitude_noisy[i, j, :].max():.6f}]")

# Normalize
signal_gt_norm = magnitude_gt[i, j, :] / (magnitude_gt[i, j, 0] + 1e-10)
signal_noisy_norm = magnitude_noisy[i, j, :] / (magnitude_noisy[i, j, 0] + 1e-10)

print(f"\nNormalize edilmiş ground truth [{i},{j}]:")
print(f"  Values: {signal_gt_norm}")

print(f"\nNormalize edilmiş gürültülü [{i},{j}]:")
print(f"  Values: {signal_noisy_norm}")

# IVIM modelden teorik sinyal oluştur
print("\n3. Teorik IVIM sinyal (GT parametrelerle)...")
theoretical_signal = ivim_fit_func(b_values, f_gt[i,j], Dt_gt[i,j], Dstar_gt[i,j])
print(f"  Values: {theoretical_signal}")

# NLLS fit - Normalize edilmiş sinyal ile
print("\n4. NLLS Fitting (normalized noisy signal)...")
try:
    fitdata, _ = curve_fit(
        ivim_fit_func,
        b_values,
        signal_noisy_norm,
        p0=[0.15, 1.5, 8],
        bounds=([0, 0, 0], [1, 2.9, 60]),
        method='trf',
        maxfev=5000
    )
    print(f"  Success: f={fitdata[0]:.4f}, Dt={fitdata[1]:.4f}, D*={fitdata[2]:.4f}")
except Exception as e:
    print(f"  Failed: {e}")

# NLLS fit - Ground truth sinyal ile (sanity check)
print("\n5. NLLS Fitting (normalized GT signal - sanity check)...")
try:
    fitdata_gt, _ = curve_fit(
        ivim_fit_func,
        b_values,
        signal_gt_norm,
        p0=[0.15, 1.5, 8],
        bounds=([0, 0, 0], [1, 2.9, 60]),
        method='trf',
        maxfev=5000
    )
    print(f"  Success: f={fitdata_gt[0]:.4f}, Dt={fitdata_gt[1]:.4f}, D*={fitdata_gt[2]:.4f}")
    print(f"  Error: Δf={abs(fitdata_gt[0]-f_gt[i,j]):.4f}, ΔDt={abs(fitdata_gt[1]-Dt_gt[i,j]):.4f}, ΔD*={abs(fitdata_gt[2]-Dstar_gt[i,j]):.4f}")
except Exception as e:
    print(f"  Failed: {e}")

# Problem teşhisi: Challenge verisindeki IVIM modeli farklı mı?
print("\n6. Challenge IVIM modeli kontrol...")
print("   Challenge'daki Dt ve Dstar değerleri mm²/s cinsinden (×10⁻³ değil)")
print("   Ama model denklemi: S = (1-f)*exp(-b*Dt) + f*exp(-b*Dstar)")
print("   b değerleri s/mm² cinsinden")
print("   Bu durumda Dt ve Dstar mm²/s cinsinden olmalı (×10⁻³ ile)")

# Alternative: b/1000 yapmadan dene
def ivim_fit_func_alt(b, f, Dt, Dstar):
    """IVIM model - b/1000 olmadan (challenge format)"""
    return (1 - f) * np.exp(-b * Dt) + f * np.exp(-b * Dstar)

print("\n7. NLLS Fitting (Alternative model - b/1000 olmadan)...")
# Bu durumda GT parametreleri de /1000 olmalı
f_gt_orig = ivim_params[i, j, 0]
Dt_gt_orig = ivim_params[i, j, 1]
Dstar_gt_orig = ivim_params[i, j, 2]

print(f"  GT (original units): f={f_gt_orig:.6f}, Dt={Dt_gt_orig:.6f}, D*={Dstar_gt_orig:.6f}")

try:
    fitdata_alt, _ = curve_fit(
        ivim_fit_func_alt,
        b_values,
        signal_noisy_norm,
        p0=[0.15, 0.0015, 0.008],
        bounds=([0, 0, 0], [1, 0.0029, 0.06]),
        method='trf',
        maxfev=5000
    )
    print(f"  Success: f={fitdata_alt[0]:.6f}, Dt={fitdata_alt[1]:.6f}, D*={fitdata_alt[2]:.6f}")
    print(f"  Error: Δf={abs(fitdata_alt[0]-f_gt_orig):.6f}, ΔDt={abs(fitdata_alt[1]-Dt_gt_orig):.6f}, ΔD*={abs(fitdata_alt[2]-Dstar_gt_orig):.6f}")
except Exception as e:
    print(f"  Failed: {e}")

# GT sinyalle de dene
print("\n8. NLLS Fitting (Alternative model - GT signal)...")
try:
    fitdata_alt_gt, _ = curve_fit(
        ivim_fit_func_alt,
        b_values,
        signal_gt_norm,
        p0=[0.15, 0.0015, 0.008],
        bounds=([0, 0, 0], [1, 0.0029, 0.06]),
        method='trf',
        maxfev=5000
    )
    print(f"  Success: f={fitdata_alt_gt[0]:.6f}, Dt={fitdata_alt_gt[1]:.6f}, D*={fitdata_alt_gt[2]:.6f}")
    print(f"  Error: Δf={abs(fitdata_alt_gt[0]-f_gt_orig):.6f}, ΔDt={abs(fitdata_alt_gt[1]-Dt_gt_orig):.6f}, ΔD*={abs(fitdata_alt_gt[2]-Dstar_gt_orig):.6f}")
except Exception as e:
    print(f"  Failed: {e}")

# Görselleştirme
print("\n9. Görselleştirme...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(b_values, signal_gt_norm, 'o-', label='GT Signal (normalized)', markersize=8)
plt.plot(b_values, signal_noisy_norm, 's-', label='Noisy Signal (normalized)', markersize=6)
plt.plot(b_values, theoretical_signal, '^-', label='Theoretical IVIM', markersize=6)
plt.xlabel('b-value (s/mm²)')
plt.ylabel('Normalized Signal')
plt.title(f'Voxel [{i},{j}] - Signals')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(b_values, signal_gt_norm, 'o-', label='GT Signal', markersize=8)
plt.semilogy(b_values, signal_noisy_norm, 's-', label='Noisy Signal', markersize=6)
plt.semilogy(b_values, theoretical_signal, '^-', label='Theoretical IVIM', markersize=6)
plt.xlabel('b-value (s/mm²)')
plt.ylabel('Normalized Signal (log scale)')
plt.title(f'Voxel [{i},{j}] - Log Scale')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('debug_nlls_signals.png', dpi=150)
print("  Saved: debug_nlls_signals.png")

print("\n" + "="*60)
print("SONUÇ:")
print("="*60)
print("Challenge verisindeki Dt ve Dstar değerleri mm²/s cinsinden")
print("IVIM modeli: S = (1-f)*exp(-b*Dt) + f*exp(-b*Dstar)")
print("PIA ve utils.py'deki hybrid_fit ise /1000 yapıyor")
print("Bu nedenle GT değerlerini *1000 ile çarptık")
print("Ama fitting için doğru birim kullanmalıyız!")
