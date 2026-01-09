"""
PIA Model Eğitimi - İyileştirilmiş Loss Function
- Hybrid Loss: MSE + MAE + Correlation
- Parametreye özel ağırlıklar
- b-değer ağırlıklandırma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIA import PIA
from utils import get_batch_optimized
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_STEPS = 100000
NOISE_STD = 0.02
SAVE_INTERVAL = 10000

# Düzeltilmiş parametre aralıkları (Challenge verisine göre)
F_MEAN = 0.15      # Challenge'da max ~0.31
F_DELTA = 0.15     # [0, 0.3] aralığı
DT_MEAN = 1.0      # Challenge'da max ~2.1
DT_DELTA = 1.0     # [0, 2.0] aralığı
DSTAR_MEAN = 25.0  # Challenge'da max ~59
DSTAR_DELTA = 25.0 # [0, 50] aralığı

# b-değerleri
b_values = [0, 5, 50, 100, 200, 500, 800, 1000]


def correlation_loss(pred, target):
    """
    Pearson korelasyon kaybı (1 - korelasyon)
    Negatif korelasyon sorununu çözmek için
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    covariance = (pred_centered * target_centered).mean()
    pred_std = torch.sqrt((pred_centered ** 2).mean() + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).mean() + 1e-8)
    
    correlation = covariance / (pred_std * target_std + 1e-8)
    
    return 1.0 - correlation


def weighted_signal_loss(pred_signal, true_signal, weights=None):
    """
    b-değerine göre ağırlıklı sinyal loss
    Düşük b-değerleri D* için kritik
    """
    if weights is None:
        # Düşük b'de yüksek ağırlık
        b_tensor = torch.tensor(b_values, dtype=torch.float32, device=pred_signal.device)
        weights = 5.0 - 4.0 * (b_tensor / b_tensor.max())
        weights = weights / weights.sum()
    
    # Her b-değeri için ağırlıklı MSE
    mse_per_b = ((pred_signal - true_signal) ** 2).mean(dim=0)
    weighted_mse = (weights * mse_per_b).sum()
    
    return weighted_mse


def hybrid_ivim_loss(model_output, true_data, 
                     signal_weight=1.0,
                     param_mse_weight=0.5,
                     param_mae_weight=0.3,
                     param_corr_weight=0.2,
                     f_weight=2.0,
                     Dt_weight=1.0,
                     Dstar_weight=3.0):
    """
    Çok bileşenli loss function
    
    Args:
        model_output: (pred_signal, input_signal, pred_f, pred_Dt, pred_Dstar)
        true_data: (true_signal, true_f, true_Dt, true_Dstar)
    """
    pred_signal, _, pred_f, pred_Dt, pred_Dstar = model_output
    true_signal, true_f, true_Dt, true_Dstar = true_data
    
    # 1. Sinyal rekonstrüksiyon kaybı (ağırlıklı)
    signal_loss = weighted_signal_loss(pred_signal, true_signal)
    
    # 2. Parametre MSE kaybı (ağırlıklı)
    param_mse = (f_weight * F.mse_loss(pred_f, true_f) +
                 Dt_weight * F.mse_loss(pred_Dt, true_Dt) +
                 Dstar_weight * F.mse_loss(pred_Dstar, true_Dstar))
    param_mse = param_mse / (f_weight + Dt_weight + Dstar_weight)
    
    # 3. Parametre MAE kaybı (ağırlıklı)
    param_mae = (f_weight * F.l1_loss(pred_f, true_f) +
                 Dt_weight * F.l1_loss(pred_Dt, true_Dt) +
                 Dstar_weight * F.l1_loss(pred_Dstar, true_Dstar))
    param_mae = param_mae / (f_weight + Dt_weight + Dstar_weight)
    
    # 4. Korelasyon kaybı (negatif korelasyonu önlemek için)
    corr_f = correlation_loss(pred_f, true_f)
    corr_Dt = correlation_loss(pred_Dt, true_Dt)
    corr_Dstar = correlation_loss(pred_Dstar, true_Dstar)
    
    # f ve D* için daha yüksek ağırlık (sorunlu parametreler)
    param_corr = (f_weight * corr_f + 
                  Dt_weight * corr_Dt + 
                  Dstar_weight * corr_Dstar)
    param_corr = param_corr / (f_weight + Dt_weight + Dstar_weight)
    
    # Toplam loss
    total_loss = (signal_weight * signal_loss +
                  param_mse_weight * param_mse +
                  param_mae_weight * param_mae +
                  param_corr_weight * param_corr)
    
    # Detaylı loss bileşenleri
    loss_dict = {
        'total': total_loss,
        'signal': signal_loss,
        'param_mse': param_mse,
        'param_mae': param_mae,
        'param_corr': param_corr,
        'corr_f': corr_f,
        'corr_Dt': corr_Dt,
        'corr_Dstar': corr_Dstar
    }
    
    return total_loss, loss_dict


def train():
    """Ana eğitim fonksiyonu"""
    
    # Model oluştur (düzeltilmiş parametrelerle)
    model = PIA(
        number_of_signals=8,
        f_mean=F_MEAN,
        f_delta=F_DELTA,
        Dt_mean=DT_MEAN,
        Dt_delta=DT_DELTA,
        Dstar_mean=DSTAR_MEAN,
        Dstar_delta=DSTAR_DELTA,
        b_values=b_values,
        hidden_dims=[32, 64, 128, 256, 512],
        predictor_depth=6,
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS)
    
    # Kayıt klasörü
    save_dir = 'pia_runs/exp4_improved_loss'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Training PIA with Improved Loss Function")
    print("="*60)
    print(f"Model parameters:")
    print(f"  f range: [{F_MEAN - F_DELTA:.2f}, {F_MEAN + F_DELTA:.2f}]")
    print(f"  Dt range: [{DT_MEAN - DT_DELTA:.2f}, {DT_MEAN + DT_DELTA:.2f}]")
    print(f"  Dstar range: [{DSTAR_MEAN - DSTAR_DELTA:.2f}, {DSTAR_MEAN + DSTAR_DELTA:.2f}]")
    print(f"\nLoss components:")
    print(f"  Signal + Param MSE + Param MAE + Param Correlation")
    print(f"  Parameter weights: f={2.0}x, Dt={1.0}x, D*={3.0}x")
    print(f"\nTraining for {NUM_STEPS} steps with batch size {BATCH_SIZE}")
    print("="*60 + "\n")
    
    # Eğitim loop
    best_loss = float('inf')
    
    for step in range(1, NUM_STEPS + 1):
        # Batch oluştur
        signals, f_true, Dt_true, Dstar_true, clean_signal = get_batch_optimized(
            batch_size=BATCH_SIZE,
            noise_sdt=NOISE_STD,
            normalize_b0=True
        )
        
        signals = signals.to(device)
        f_true = f_true.to(device)
        Dt_true = Dt_true.to(device)
        Dstar_true = Dstar_true.to(device)
        clean_signal = clean_signal.to(device)
        
        # Forward pass
        model_output = model(signals)
        true_data = (clean_signal, f_true, Dt_true, Dstar_true)
        
        # Loss hesapla
        loss, loss_dict = hybrid_ivim_loss(model_output, true_data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step}/{NUM_STEPS} | "
                  f"Loss: {loss_dict['total']:.6f} | "
                  f"Signal: {loss_dict['signal']:.6f} | "
                  f"MSE: {loss_dict['param_mse']:.6f} | "
                  f"MAE: {loss_dict['param_mae']:.6f} | "
                  f"Corr: {loss_dict['param_corr']:.6f}")
            
            print(f"  Correlations - f: {1-loss_dict['corr_f'].item():.4f}, "
                  f"Dt: {1-loss_dict['corr_Dt'].item():.4f}, "
                  f"D*: {1-loss_dict['corr_Dstar'].item():.4f}")
        
        # Model kaydet
        if step % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_step{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'loss_dict': {k: v.item() if torch.is_tensor(v) else v 
                             for k, v in loss_dict.items()}
            }, checkpoint_path)
            print(f"\n[Saved] Checkpoint at step {step}: {checkpoint_path}\n")
        
        # En iyi model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'loss': best_loss
            }, best_path)
    
    # Final model
    final_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'step': NUM_STEPS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
    }, final_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final model saved to: {final_path}")
    print("="*60)


if __name__ == '__main__':
    train()
