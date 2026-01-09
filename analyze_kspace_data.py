"""
K-space verisinden IVIM parametre tahmini ve karşılaştırma
Hybrid Fit (NLLS) vs Eğitilmiş PIA Modelleri
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import fft
from scipy.stats import spearmanr
import os
from PIA import PIA
from utils import hybrid_fit

# Device ayarı
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# b-değerleri
b_values = [0, 5, 50, 100, 200, 500, 800, 1000]

def kspace_to_image(kspace_data):
    """
    k-space verisini image domain'e dönüştürür (inverse FFT)
    
    Args:
        kspace_data: (H, W, N_b) kompleks k-space verisi
    
    Returns:
        image_data: (H, W, N_b) kompleks image domain verisi
    """
    # 2D inverse FFT (spatial dimensions için)
    image_data = fft.ifft2(kspace_data, axes=(0, 1))
    # FFT shift (center'a al)
    image_data = fft.ifftshift(image_data, axes=(0, 1))
    return image_data


def load_data(patient_id='0001'):
    """Challenge verisini yükler"""
    data_dir = 'data'
    
    # Ground truth parametreler
    ivim_params = np.load(os.path.join(data_dir, f'{patient_id}_IVIMParam.npy'))
    f_gt = ivim_params[:, :, 0]
    # Challenge verisi mm²/s cinsinden, PIA/hybrid_fit ×10⁻³ mm²/s kullanıyor
    # Bu yüzden Dt ve Dstar'ı 1000 ile çarpıyoruz
    Dt_gt = ivim_params[:, :, 1] * 1000
    Dstar_gt = ivim_params[:, :, 2] * 1000
    
    # Ground truth DWI sinyalleri (kompleks)
    gt_dwis = np.load(os.path.join(data_dir, f'{patient_id}_gtDWIs.npy'))
    
    # Gürültülü k-space verisi
    noisy_kspace = np.load(os.path.join(data_dir, f'{patient_id}_NoisyDWIk.npy'))
    
    # Doku tipleri
    tissue_type = np.load(os.path.join(data_dir, f'{patient_id}_TissueType.npy'))
    
    return {
        'f_gt': f_gt,
        'Dt_gt': Dt_gt,
        'Dstar_gt': Dstar_gt,
        'gt_dwis': gt_dwis,
        'noisy_kspace': noisy_kspace,
        'tissue_type': tissue_type
    }


def reconstruct_from_kspace(noisy_kspace):
    """
    k-space'den magnitude sinyali elde eder ve normalize eder
    
    Returns:
        magnitude_signal: (H, W, N_b) normalized magnitude değerleri
    """
    # k-space -> image domain (doğru yöntem: ifft2)
    image_data = fft.ifft2(noisy_kspace, axes=(0, 1))
    
    # FFT normalization scaling (numpy ifft2 1/N ile normalize eder)
    H, W = noisy_kspace.shape[:2]
    image_data = image_data * (H * W)
    
    # Magnitude (Rician gürültü varsayımı)
    magnitude_signal = np.abs(image_data)
    
    # Normalize by b=0 (PIA ve NLLS için gerekli)
    magnitude_signal = magnitude_signal / (magnitude_signal[:, :, [0]] + 1e-8)
    
    return magnitude_signal


def estimate_params_nlls(magnitude_signal, mask=None):
    """
    Hybrid fit (NLLS) ile parametre tahmini
    
    Args:
        magnitude_signal: (H, W, N_b) magnitude sinyalleri
        mask: Sadece mask içindeki voxelleri işle
    
    Returns:
        f_est, Dt_est, Dstar_est: (H, W) tahmin edilen parametreler
    """
    H, W, N_b = magnitude_signal.shape
    
    f_est = np.zeros((H, W))
    Dt_est = np.zeros((H, W))
    Dstar_est = np.zeros((H, W))
    
    # Her voxel için
    total_voxels = H * W
    processed = 0
    
    for i in range(H):
        for j in range(W):
            # Mask kontrolü
            if mask is not None and not mask[i, j]:
                continue
            
            # Voxel sinyali
            signal = magnitude_signal[i, j, :].reshape(1, -1)
            
            # NLLS fit
            f_v, Dt_v, Dstar_v = hybrid_fit(signal, bvals=b_values)
            
            f_est[i, j] = f_v[0]
            Dt_est[i, j] = Dt_v[0]
            Dstar_est[i, j] = Dstar_v[0]
            
            processed += 1
            if processed % 1000 == 0:
                print(f"NLLS Progress: {processed}/{total_voxels} voxels ({100*processed/total_voxels:.1f}%)")
    
    return f_est, Dt_est, Dstar_est


def estimate_params_pia(magnitude_signal, model_path, mask=None):
    """
    Eğitilmiş PIA modeli ile parametre tahmini
    
    Args:
        magnitude_signal: (H, W, N_b) magnitude sinyalleri
        model_path: Model checkpoint yolu
        mask: Sadece mask içindeki voxelleri işle
    
    Returns:
        f_est, Dt_est, Dstar_est: (H, W) tahmin edilen parametreler
    """
    H, W, N_b = magnitude_signal.shape
    
    # Model yükleme
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
    
    f_est = np.zeros((H, W))
    Dt_est = np.zeros((H, W))
    Dstar_est = np.zeros((H, W))
    
    # Batch processing için voxelleri topla
    voxel_signals = []
    voxel_coords = []
    
    for i in range(H):
        for j in range(W):
            if mask is not None and not mask[i, j]:
                continue
            voxel_signals.append(magnitude_signal[i, j, :])
            voxel_coords.append((i, j))
    
    # Batch olarak işle
    batch_size = 1024
    total_voxels = len(voxel_signals)
    
    with torch.no_grad():
        for start_idx in range(0, total_voxels, batch_size):
            end_idx = min(start_idx + batch_size, total_voxels)
            batch_signals = np.array(voxel_signals[start_idx:end_idx])
            batch_coords = voxel_coords[start_idx:end_idx]
            
            # Tensor'e çevir
            signals_tensor = torch.from_numpy(batch_signals).float().to(device)
            
            # Model tahmini
            f, Dt, Dstar = model.encode(signals_tensor)
            
            # CPU'ya al
            f_np = f.cpu().numpy().flatten()
            Dt_np = Dt.cpu().numpy().flatten()
            Dstar_np = Dstar.cpu().numpy().flatten()
            
            # Sonuçları yerleştir
            for idx, (i, j) in enumerate(batch_coords):
                f_est[i, j] = f_np[idx]
                Dt_est[i, j] = Dt_np[idx]
                Dstar_est[i, j] = Dstar_np[idx]
            
            if end_idx % 5000 == 0:
                print(f"PIA Progress: {end_idx}/{total_voxels} voxels ({100*end_idx/total_voxels:.1f}%)")
    
    return f_est, Dt_est, Dstar_est


def compute_metrics(pred, gt, mask=None):
    """RMSE, NRMSE, MAE ve Spearman korelasyonu hesaplar"""
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    else:
        pred = pred.flatten()
        gt = gt.flatten()
    
    # NaN ve Inf kontrolü
    valid_mask = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    
    # RMSE ve NRMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    nrmse = rmse / (np.max(gt) - np.min(gt) + 1e-8)
    
    # MAE
    mae = np.mean(np.abs(pred - gt))
    
    # Spearman korelasyonu
    if len(pred) > 1:
        spearman_corr, _ = spearmanr(pred, gt)
    else:
        spearman_corr = np.nan
    
    return rmse, nrmse, mae, spearman_corr


def visualize_results(data, nlls_results, pia_results, model_name, save_path='results'):
    """Sonuçları görselleştirir"""
    os.makedirs(save_path, exist_ok=True)
    
    # Mask oluştur (sıfır olmayan voxeller)
    mask = data['tissue_type'] > 0
    
    # Metrikler
    print("\n" + "="*60)
    print(f"Model: {model_name}")
    print("="*60)
    
    print("\nNLLS (Hybrid Fit) Metrics:")
    for param_name, pred, gt in [
        ('f', nlls_results['f'], data['f_gt']),
        ('Dt', nlls_results['Dt'], data['Dt_gt']),
        ('Dstar', nlls_results['Dstar'], data['Dstar_gt'])
    ]:
        rmse, nrmse, mae, spearman = compute_metrics(pred, gt, mask)
        print(f"  {param_name:6s}: RMSE={rmse:.6f}, NRMSE={nrmse:.4f}, MAE={mae:.6f}, Spearman={spearman:.4f}")
    
    print("\nPIA Model Metrics:")
    for param_name, pred, gt in [
        ('f', pia_results['f'], data['f_gt']),
        ('Dt', pia_results['Dt'], data['Dt_gt']),
        ('Dstar', pia_results['Dstar'], data['Dstar_gt'])
    ]:
        rmse, nrmse, mae, spearman = compute_metrics(pred, gt, mask)
        print(f"  {param_name:6s}: RMSE={rmse:.6f}, NRMSE={nrmse:.4f}, MAE={mae:.6f}, Spearman={spearman:.4f}")
    
    # Görselleştirme
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    params = [
        ('f', data['f_gt'], nlls_results['f'], pia_results['f']),
        ('Dt', data['Dt_gt'], nlls_results['Dt'], pia_results['Dt']),
        ('D*', data['Dstar_gt'], nlls_results['Dstar'], pia_results['Dstar'])
    ]
    
    for row, (param_name, gt, nlls_pred, pia_pred) in enumerate(params):
        # Ground Truth
        im0 = axes[row, 0].imshow(gt, cmap='hot')
        axes[row, 0].set_title(f'{param_name} - Ground Truth')
        axes[row, 0].axis('off')
        plt.colorbar(im0, ax=axes[row, 0])
        
        # NLLS
        im1 = axes[row, 1].imshow(nlls_pred, cmap='hot')
        axes[row, 1].set_title(f'{param_name} - NLLS')
        axes[row, 1].axis('off')
        plt.colorbar(im1, ax=axes[row, 1])
        
        # PIA
        im2 = axes[row, 2].imshow(pia_pred, cmap='hot')
        axes[row, 2].set_title(f'{param_name} - PIA Model')
        axes[row, 2].axis('off')
        plt.colorbar(im2, ax=axes[row, 2])
        
        # Error maps
        nlls_error = np.abs(nlls_pred - gt)
        pia_error = np.abs(pia_pred - gt)
        
        # NLLS Error
        im3 = axes[row, 3].imshow(nlls_error, cmap='Reds', vmin=0)
        axes[row, 3].set_title(f'{param_name} - NLLS Error')
        axes[row, 3].axis('off')
        plt.colorbar(im3, ax=axes[row, 3])
    
    plt.suptitle(f'IVIM Parameter Estimation - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'comparison_{model_name}.png'), dpi=150)
    print(f"\nVisualization saved to: {os.path.join(save_path, f'comparison_{model_name}.png')}")
    plt.close()


def main():
    """Ana analiz pipeline"""
    
    print("="*60)
    print("K-space IVIM Parameter Estimation")
    print("="*60)
    
    # Veri yükleme
    print("\n1. Loading data...")
    patient_id = '0001'
    data = load_data(patient_id)
    print(f"   Patient {patient_id} loaded")
    print(f"   Image size: {data['noisy_kspace'].shape[:2]}")
    print(f"   Number of b-values: {data['noisy_kspace'].shape[2]}")
    
    # k-space'den rekonstrüksiyon
    print("\n2. Reconstructing from k-space...")
    magnitude_signal = reconstruct_from_kspace(data['noisy_kspace'])
    print(f"   Magnitude signal range: [{magnitude_signal.min():.4f}, {magnitude_signal.max():.4f}]")
    
    # Mask (sadece tissue olan voxelleri işle)
    mask = data['tissue_type'] > 0
    print(f"   Processing {mask.sum()} voxels (out of {mask.size})")
    
    # NLLS (Hybrid Fit)
    print("\n3. NLLS (Hybrid Fit) estimation...")
    f_nlls, Dt_nlls, Dstar_nlls = estimate_params_nlls(magnitude_signal, mask)
    nlls_results = {
        'f': f_nlls,
        'Dt': Dt_nlls,
        'Dstar': Dstar_nlls
    }
    print("   NLLS estimation completed")
    
    # Eğitilmiş modelleri bul
    print("\n4. Finding trained models...")
    model_paths = []
    
    # exp2 model
    exp2_path = 'pia_runs/exp2_balanced_weighted/best_high_noise.pt'
    if os.path.exists(exp2_path):
        model_paths.append(('exp2_best_high_noise', exp2_path))
    
    exp2_final = 'pia_runs/exp2_balanced_weighted/final_model.pt'
    if os.path.exists(exp2_final):
        model_paths.append(('exp2_final', exp2_final))
    
    # exp3 checkpoints
    exp3_50k = 'pia_runs/exp3_finetune_balanced/checkpoint_step50000.pt'
    if os.path.exists(exp3_50k):
        model_paths.append(('exp3_step50k', exp3_50k))
    
    exp3_62k = 'pia_runs/exp3_finetune_balanced/checkpoint_step62500.pt'
    if os.path.exists(exp3_62k):
        model_paths.append(('exp3_step62k', exp3_62k))
    
    print(f"   Found {len(model_paths)} trained models")
    
    # Her model için analiz
    for model_name, model_path in model_paths:
        print(f"\n5. Analyzing model: {model_name}")
        print("-" * 60)
        
        # PIA model tahmini
        f_pia, Dt_pia, Dstar_pia = estimate_params_pia(magnitude_signal, model_path, mask)
        pia_results = {
            'f': f_pia,
            'Dt': Dt_pia,
            'Dstar': Dstar_pia
        }
        
        # Görselleştirme ve metrikler
        visualize_results(data, nlls_results, pia_results, model_name)
    
    print("\n" + "="*60)
    print("Analysis completed!")
    print("="*60)


if __name__ == '__main__':
    main()
