"""
Eğitilmiş PIA Modelini Challenge Verisiyle Fine-tune Etme

Strateji:
1. Yüksek SNR (signal-to-noise ratio) voxelleri seç
2. Bu voxellerle küçük batch'ler oluştur
3. Düşük learning rate ile fine-tune
4. Hybrid loss kullan
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import fft
from scipy.stats import spearmanr
import os
from PIA import PIA

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
FINE_TUNE_LR = 5e-5  # Çok düşük learning rate
BATCH_SIZE = 256
NUM_EPOCHS = 50
SNR_THRESHOLD = 5.0  # SNR > 5 olan voxelleri kullan

# b-değerleri
b_values = [0, 5, 50, 100, 200, 500, 800, 1000]


def load_challenge_data(patient_id='0001'):
    """Challenge verisini yükle ve işle"""
    data_dir = 'data'
    
    # Ground truth parametreler
    ivim_params = np.load(os.path.join(data_dir, f'{patient_id}_IVIMParam.npy'))
    f_gt = ivim_params[:, :, 0]
    Dt_gt = ivim_params[:, :, 1] * 1000  # Birim dönüşümü
    Dstar_gt = ivim_params[:, :, 2] * 1000
    
    # k-space verisi
    noisy_kspace = np.load(os.path.join(data_dir, f'{patient_id}_NoisyDWIk.npy'))
    
    # Doku tipleri
    tissue_type = np.load(os.path.join(data_dir, f'{patient_id}_TissueType.npy'))
    
    # k-space'den rekonstrüksiyon
    image_data = fft.ifft2(noisy_kspace, axes=(0, 1))
    H, W = noisy_kspace.shape[:2]
    image_data = image_data * (H * W)  # FFT scaling
    
    magnitude_signal = np.abs(image_data)
    
    # Normalize by b=0
    magnitude_normalized = magnitude_signal / (magnitude_signal[:, :, [0]] + 1e-8)
    
    return {
        'signals': magnitude_normalized,
        'f_gt': f_gt,
        'Dt_gt': Dt_gt,
        'Dstar_gt': Dstar_gt,
        'tissue_type': tissue_type
    }


def compute_snr(signals):
    """
    Signal-to-Noise Ratio hesapla
    SNR = b=0 signal / std(other signals)
    """
    b0_signal = signals[:, :, 0]
    noise_std = signals[:, :, 1:].std(axis=2)
    snr = b0_signal / (noise_std + 1e-8)
    return snr


def select_high_quality_voxels(data, snr_threshold=5.0):
    """
    Yüksek kaliteli voxelleri seç
    """
    snr = compute_snr(data['signals'])
    
    # Mask: yüksek SNR + tissue olan voxeller
    high_quality_mask = (snr > snr_threshold) & (data['tissue_type'] > 0)
    
    # İndeksleri al
    indices = np.where(high_quality_mask)
    
    # Voxelleri topla
    voxel_signals = data['signals'][indices]
    voxel_f = data['f_gt'][indices]
    voxel_Dt = data['Dt_gt'][indices]
    voxel_Dstar = data['Dstar_gt'][indices]
    
    return voxel_signals, voxel_f, voxel_Dt, voxel_Dstar, len(indices[0])


def correlation_loss(pred, target):
    """Pearson korelasyon kaybı"""
    pred_mean = pred.mean()
    target_mean = target.mean()
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    covariance = (pred_centered * target_centered).mean()
    pred_std = torch.sqrt((pred_centered ** 2).mean() + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).mean() + 1e-8)
    
    correlation = covariance / (pred_std * target_std + 1e-8)
    
    return 1.0 - correlation


def hybrid_param_loss(pred_f, true_f, pred_Dt, true_Dt, pred_Dstar, true_Dstar):
    """
    Hibrit parametre loss: MSE + MAE + Correlation
    f ve D* için daha yüksek ağırlık (sorunlu parametreler)
    """
    # MSE
    mse_f = F.mse_loss(pred_f, true_f)
    mse_Dt = F.mse_loss(pred_Dt, true_Dt)
    mse_Dstar = F.mse_loss(pred_Dstar, true_Dstar)
    
    # MAE
    mae_f = F.l1_loss(pred_f, true_f)
    mae_Dt = F.l1_loss(pred_Dt, true_Dt)
    mae_Dstar = F.l1_loss(pred_Dstar, true_Dstar)
    
    # Correlation
    corr_f = correlation_loss(pred_f, true_f)
    corr_Dt = correlation_loss(pred_Dt, true_Dt)
    corr_Dstar = correlation_loss(pred_Dstar, true_Dstar)
    
    # Ağırlıklı toplam (f ve D* için 2x, Dt için 1x)
    total_loss = (
        2.0 * (0.5 * mse_f + 0.3 * mae_f + 0.2 * corr_f) +
        1.0 * (0.5 * mse_Dt + 0.3 * mae_Dt + 0.2 * corr_Dt) +
        3.0 * (0.5 * mse_Dstar + 0.3 * mae_Dstar + 0.2 * corr_Dstar)
    ) / 6.0
    
    return total_loss, {
        'mse_f': mse_f, 'mae_f': mae_f, 'corr_f': corr_f,
        'mse_Dt': mse_Dt, 'mae_Dt': mae_Dt, 'corr_Dt': corr_Dt,
        'mse_Dstar': mse_Dstar, 'mae_Dstar': mae_Dstar, 'corr_Dstar': corr_Dstar
    }


def evaluate_model(model, data):
    """Model performansını değerlendir"""
    model.eval()
    
    signals = torch.from_numpy(data['signals'].reshape(-1, 8)).float().to(device)
    f_gt = data['f_gt'].flatten()
    Dt_gt = data['Dt_gt'].flatten()
    Dstar_gt = data['Dstar_gt'].flatten()
    
    with torch.no_grad():
        # Batch olarak tahmin
        batch_size = 2048
        f_pred_list = []
        Dt_pred_list = []
        Dstar_pred_list = []
        
        for i in range(0, len(signals), batch_size):
            batch = signals[i:i+batch_size]
            f_p, Dt_p, Dstar_p = model.encode(batch)
            f_pred_list.append(f_p.cpu().numpy())
            Dt_pred_list.append(Dt_p.cpu().numpy())
            Dstar_pred_list.append(Dstar_p.cpu().numpy())
        
        f_pred = np.concatenate(f_pred_list).flatten()
        Dt_pred = np.concatenate(Dt_pred_list).flatten()
        Dstar_pred = np.concatenate(Dstar_pred_list).flatten()
    
    # Metrikler
    mask = (data['tissue_type'].flatten() > 0) & np.isfinite(f_pred) & np.isfinite(f_gt)
    
    metrics = {}
    for name, pred, gt in [('f', f_pred, f_gt), ('Dt', Dt_pred, Dt_gt), ('Dstar', Dstar_pred, Dstar_gt)]:
        valid_pred = pred[mask]
        valid_gt = gt[mask]
        
        rmse = np.sqrt(np.mean((valid_pred - valid_gt) ** 2))
        mae = np.mean(np.abs(valid_pred - valid_gt))
        spearman, _ = spearmanr(valid_pred, valid_gt)
        
        metrics[name] = {'rmse': rmse, 'mae': mae, 'spearman': spearman}
    
    model.train()
    return metrics


def fine_tune(model_path, patient_id='0001', output_dir='pia_runs/exp4_finetuned'):
    """
    Ana fine-tuning fonksiyonu
    """
    
    print("\n" + "="*60)
    print("Fine-tuning PIA Model with Challenge Data")
    print("="*60)
    
    # Model yükle
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
    
    print(f"Loaded model from: {model_path}")
    
    # Challenge verisi yükle
    print(f"\nLoading challenge data (patient {patient_id})...")
    data = load_challenge_data(patient_id)
    
    # Yüksek kaliteli voxelleri seç
    print(f"Selecting high-quality voxels (SNR > {SNR_THRESHOLD})...")
    voxel_signals, voxel_f, voxel_Dt, voxel_Dstar, num_voxels = select_high_quality_voxels(
        data, snr_threshold=SNR_THRESHOLD
    )
    
    print(f"Selected {num_voxels} high-quality voxels out of {data['tissue_type'].size}")
    print(f"  f range: [{voxel_f.min():.4f}, {voxel_f.max():.4f}]")
    print(f"  Dt range: [{voxel_Dt.min():.4f}, {voxel_Dt.max():.4f}]")
    print(f"  D* range: [{voxel_Dstar.min():.4f}, {voxel_Dstar.max():.4f}]")
    
    # Baseline performans
    print("\nEvaluating baseline performance...")
    baseline_metrics = evaluate_model(model, data)
    print("Baseline metrics:")
    for param in ['f', 'Dt', 'Dstar']:
        m = baseline_metrics[param]
        print(f"  {param:5s}: RMSE={m['rmse']:.6f}, MAE={m['mae']:.6f}, Spearman={m['spearman']:.4f}")
    
    # Optimizer (düşük learning rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    
    # Output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fine-tuning loop
    print(f"\nFine-tuning for {NUM_EPOCHS} epochs...")
    print("="*60)
    
    best_metric = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Voxelleri karıştır
        indices = np.random.permutation(num_voxels)
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_voxels, BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            
            # Batch oluştur
            signals = torch.from_numpy(voxel_signals[batch_indices]).float().to(device)
            f_true = torch.from_numpy(voxel_f[batch_indices]).float().unsqueeze(1).to(device)
            Dt_true = torch.from_numpy(voxel_Dt[batch_indices]).float().unsqueeze(1).to(device)
            Dstar_true = torch.from_numpy(voxel_Dstar[batch_indices]).float().unsqueeze(1).to(device)
            
            # Forward
            f_pred, Dt_pred, Dstar_pred = model.encode(signals)
            
            # Loss
            loss, loss_dict = hybrid_param_loss(f_pred, f_true, Dt_pred, Dt_true, Dstar_pred, Dstar_true)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Epoch sonuçları
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate_model(model, data)
            print(f"\nEpoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.6f}")
            for param in ['f', 'Dt', 'Dstar']:
                m = metrics[param]
                print(f"  {param:5s}: RMSE={m['rmse']:.6f}, MAE={m['mae']:.6f}, Spearman={m['spearman']:.4f}")
            
            # En iyi model kaydet
            avg_rmse = (metrics['f']['rmse'] + metrics['Dt']['rmse'] + metrics['Dstar']['rmse']) / 3
            if avg_rmse < best_metric:
                best_metric = avg_rmse
                best_path = os.path.join(output_dir, f'best_finetuned_{patient_id}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, best_path)
                print(f"  [Saved best model] Avg RMSE: {avg_rmse:.6f}")
    
    # Final model
    final_path = os.path.join(output_dir, f'final_finetuned_{patient_id}.pt')
    final_metrics = evaluate_model(model, data)
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'metrics': final_metrics
    }, final_path)
    
    print("\n" + "="*60)
    print("Fine-tuning completed!")
    print(f"Best model: {best_path}")
    print(f"Final model: {final_path}")
    
    print("\nFinal metrics:")
    for param in ['f', 'Dt', 'Dstar']:
        m = final_metrics[param]
        print(f"  {param:5s}: RMSE={m['rmse']:.6f}, MAE={m['mae']:.6f}, Spearman={m['spearman']:.4f}")
    
    print("\nImprovement vs baseline:")
    for param in ['f', 'Dt', 'Dstar']:
        base = baseline_metrics[param]
        final = final_metrics[param]
        rmse_improve = 100 * (base['rmse'] - final['rmse']) / base['rmse']
        mae_improve = 100 * (base['mae'] - final['mae']) / base['mae']
        spear_improve = final['spearman'] - base['spearman']
        print(f"  {param:5s}: RMSE {rmse_improve:+.1f}%, MAE {mae_improve:+.1f}%, Spearman {spear_improve:+.3f}")
    
    print("="*60)
    
    return model


if __name__ == '__main__':
    # En iyi mevcut modeli fine-tune et
    MODEL_PATH = 'pia_runs/exp3_finetune_balanced/checkpoint_step62500.pt'
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Available models:")
        for root, dirs, files in os.walk('pia_runs'):
            for file in files:
                if file.endswith('.pt'):
                    print(f"  {os.path.join(root, file)}")
    else:
        fine_tune(MODEL_PATH, patient_id='0001')
