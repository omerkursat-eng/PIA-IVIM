# PIA Model İyileştirme Planı

## 🎯 Tespit Edilen Sorunlar

### 1. f Parametresi
- ✅ RMSE iyi (0.278 vs 0.325)
- ❌ Negatif Spearman korelasyonu (-0.37)
- ❌ MAE yüksek (0.217 vs 0.179)

### 2. D* Parametresi  
- ≈ RMSE benzer (29.8 vs 29.3)
- ❌ Negatif Spearman korelasyonu (-0.63)
- ❌ MAE çok yüksek (23.84 vs 18.60)

### 3. Dt Parametresi
- ✅✅✅ Tüm metrikler mükemmel!

## 💡 Önerilen İyileştirmeler

### Yaklaşım 1: Loss Function Geliştirme (EN KOLAY - HEMEN DENENEBİLİR)

**Problem**: Sadece MSE kullanılıyor, bu büyük hataları penalize ediyor ama küçük hataların kalitesini gözetmiyor.

**Çözümler**:

#### 1.1. Correlation Loss Ekleme
```python
def correlation_loss(pred, target):
    """Pearson korelasyon kaybı (1 - korelasyon)"""
    pred_mean = pred.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    correlation = (pred_centered * target_centered).sum(dim=0) / (
        torch.sqrt((pred_centered ** 2).sum(dim=0)) * 
        torch.sqrt((target_centered ** 2).sum(dim=0)) + 1e-8
    )
    
    return 1 - correlation.mean()
```

#### 1.2. Hybrid Loss (MSE + MAE + Correlation)
```python
def hybrid_ivim_loss(pred_signal, true_signal, pred_f, true_f, 
                     pred_Dt, true_Dt, pred_Dstar, true_Dstar, 
                     signal_weight=1.0, mse_weight=0.5, mae_weight=0.3, corr_weight=0.2):
    """
    Çok bileşenli loss function
    """
    # Sinyal rekonstrüksiyon kaybı
    signal_loss = F.mse_loss(pred_signal, true_signal)
    
    # Parametre kaybı - MSE
    param_mse = (F.mse_loss(pred_f, true_f) + 
                 F.mse_loss(pred_Dt, true_Dt) + 
                 F.mse_loss(pred_Dstar, true_Dstar)) / 3
    
    # Parametre kaybı - MAE
    param_mae = (F.l1_loss(pred_f, true_f) + 
                 F.l1_loss(pred_Dt, true_Dt) + 
                 F.l1_loss(pred_Dstar, true_Dstar)) / 3
    
    # Korelasyon kaybı
    corr_f = correlation_loss(pred_f, true_f)
    corr_Dt = correlation_loss(pred_Dt, true_Dt)
    corr_Dstar = correlation_loss(pred_Dstar, true_Dstar)
    param_corr = (corr_f + corr_Dt + corr_Dstar) / 3
    
    # Toplam loss
    total_loss = (signal_weight * signal_loss + 
                  mse_weight * param_mse + 
                  mae_weight * param_mae + 
                  corr_weight * param_corr)
    
    return total_loss
```

#### 1.3. Parametreye Özel Ağırlıklı Loss
```python
def weighted_param_loss(pred_f, true_f, pred_Dt, true_Dt, pred_Dstar, true_Dstar):
    """
    Her parametre için farklı ağırlık
    Dt zaten iyi çalışıyor, f ve D* daha fazla odak
    """
    # f için yüksek ağırlık (korelasyon sorunu)
    loss_f = F.mse_loss(pred_f, true_f) + 0.5 * F.l1_loss(pred_f, true_f)
    
    # Dt için normal ağırlık
    loss_Dt = F.mse_loss(pred_Dt, true_Dt)
    
    # D* için çok yüksek ağırlık (en sorunlu parametre)
    loss_Dstar = F.mse_loss(pred_Dstar, true_Dstar) + F.l1_loss(pred_Dstar, true_Dstar)
    
    # Ağırlıklı toplam
    total = 3.0 * loss_f + 1.0 * loss_Dt + 5.0 * loss_Dstar
    
    return total / 9.0  # Normalize
```

---

### Yaklaşım 2: Parametre Range Düzeltmesi (HIZLI FIX)

**Problem**: Challenge verisinde f max=0.31, ama model f_mean=0.5, f_delta=0.5 ([0,1] aralığı)

**Çözüm**: Model parametrelerini gerçek veri dağılımına göre ayarla

```python
# Mevcut (yanlış)
model = PIA(
    f_mean=0.5, f_delta=0.5,           # [0, 1]
    Dt_mean=1.45, Dt_delta=1.45,       # [0, 2.9]
    Dstar_mean=30, Dstar_delta=30      # [0, 60]
)

# Düzeltilmiş (challenge verisine göre)
model = PIA(
    f_mean=0.15, f_delta=0.15,         # [0, 0.3] - daha gerçekçi
    Dt_mean=1.0, Dt_delta=1.0,         # [0, 2.0] - veriye daha yakın
    Dstar_mean=25, Dstar_delta=25      # [0, 50] - biraz daha sıkı
)
```

---

### Yaklaşım 3: Gerçek Veriyle Fine-tuning (ORTA ÇABA)

**Problem**: Model sentetik veriyle eğitilmiş, gerçek challenge verisine adapte değil

**Çözüm**: Challenge verisinin temiz kısmıyla fine-tune

```python
def finetune_with_real_data(model, patient_data, epochs=50, lr=1e-5):
    """
    Gerçek veriyle fine-tuning
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Düşük gürültülü voxelleri seç (SNR > threshold)
    magnitude_signal = patient_data['magnitude_signal']
    snr = magnitude_signal[:, :, 0] / magnitude_signal[:, :, 1:].std(axis=2)
    high_snr_mask = snr > 10  # Yüksek SNR
    
    # Ground truth parametreler
    f_gt = patient_data['f_gt']
    Dt_gt = patient_data['Dt_gt']
    Dstar_gt = patient_data['Dstar_gt']
    
    for epoch in range(epochs):
        # Yüksek SNR voxellerden batch oluştur
        indices = np.where(high_snr_mask)
        batch_size = 256
        
        for i in range(0, len(indices[0]), batch_size):
            batch_indices = (indices[0][i:i+batch_size], 
                           indices[1][i:i+batch_size])
            
            signals = magnitude_signal[batch_indices]
            f_true = f_gt[batch_indices]
            Dt_true = Dt_gt[batch_indices]
            Dstar_true = Dstar_gt[batch_indices]
            
            # Forward
            signals_tensor = torch.from_numpy(signals).float().to(device)
            f_pred, Dt_pred, Dstar_pred = model.encode(signals_tensor)
            
            # Loss (hybrid)
            loss = hybrid_ivim_loss(
                model.decode(f_pred, Dt_pred, Dstar_pred), signals_tensor,
                f_pred, torch.from_numpy(f_true).float().to(device),
                Dt_pred, torch.from_numpy(Dt_true).float().to(device),
                Dstar_pred, torch.from_numpy(Dstar_true).float().to(device)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    return model
```

---

### Yaklaşım 4: b-değer Ağırlıklandırma (KOLAY)

**Problem**: D* düşük b-değerlerine duyarlı, ama tüm b-değerleri eşit ağırlıklı

**Çözüm**: Düşük b-değerlerine daha fazla ağırlık ver

```python
def weighted_signal_loss(pred_signal, true_signal, b_values):
    """
    b-değerine göre ağırlıklı loss
    Düşük b -> D* tahmini için kritik -> yüksek ağırlık
    Yüksek b -> Dt tahmini için önemli -> normal ağırlık
    """
    b_tensor = torch.tensor(b_values, device=pred_signal.device)
    
    # Ağırlıklar: düşük b'de yüksek, yüksek b'de normal
    # Örnek: b=0->5, b=5->4, b=50->3, ..., b=1000->1
    weights = 5.0 - 4.0 * (b_tensor / b_tensor.max())
    weights = weights / weights.sum()  # Normalize
    
    # Ağırlıklı MSE
    weighted_loss = (weights * (pred_signal - true_signal) ** 2).sum(dim=1).mean()
    
    return weighted_loss
```

---

### Yaklaşım 5: Two-Stage Training (ORTA-ZOR)

**Problem**: f, Dt ve D* parametreleri birbiriyle etkileşimli, aynı anda öğrenme zor

**Çözüm**: İki aşamalı eğitim

```python
def two_stage_training(model, train_loader, epochs_stage1=100, epochs_stage2=100):
    """
    Stage 1: Dt'yi optimize et (en kolay parametre)
    Stage 2: f ve D*'ı optimize et (Dt frozen)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # STAGE 1: Sadece Dt
    print("Stage 1: Training Dt...")
    for epoch in range(epochs_stage1):
        for signals, _, Dt_true, _ in train_loader:
            signals = signals.to(device)
            Dt_true = Dt_true.to(device)
            
            _, Dt_pred, _ = model.encode(signals)
            
            # Sadece Dt loss
            loss = F.mse_loss(Dt_pred, Dt_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Dt encoder parametrelerini dondur
    for param in model.Dt_predictor.parameters():
        param.requires_grad = False
    
    # STAGE 2: f ve D*
    print("Stage 2: Training f and D*...")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=5e-4
    )
    
    for epoch in range(epochs_stage2):
        for signals, f_true, Dt_true, Dstar_true in train_loader:
            signals = signals.to(device)
            f_true = f_true.to(device)
            Dstar_true = Dstar_true.to(device)
            
            f_pred, Dt_pred, Dstar_pred = model.encode(signals)
            
            # f ve D* loss (hybrid)
            loss = (F.mse_loss(f_pred, f_true) + F.l1_loss(f_pred, f_true) +
                   F.mse_loss(Dstar_pred, Dstar_true) + F.l1_loss(Dstar_pred, Dstar_true))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model
```

---

### Yaklaşım 6: Ensemble Model (ZOR AMA ETKİLİ)

**Çözüm**: Farklı loss'larla eğitilmiş modelleri birleştir

```python
def ensemble_prediction(models, signal):
    """
    Birden fazla modelin tahminlerini birleştir
    """
    predictions_f = []
    predictions_Dt = []
    predictions_Dstar = []
    
    for model in models:
        with torch.no_grad():
            f, Dt, Dstar = model.encode(signal)
            predictions_f.append(f)
            predictions_Dt.append(Dt)
            predictions_Dstar.append(Dstar)
    
    # Median al (outlier'lara karşı robust)
    f_final = torch.median(torch.stack(predictions_f), dim=0)[0]
    Dt_final = torch.median(torch.stack(predictions_Dt), dim=0)[0]
    Dstar_final = torch.median(torch.stack(predictions_Dstar), dim=0)[0]
    
    return f_final, Dt_final, Dstar_final
```

---

## 🚀 Önerilen Uygulama Sırası

### Faz 1: Hızlı Kazançlar (1-2 gün)
1. ✅ **Parametre range düzeltmesi** → Yeni model eğit
2. ✅ **Hybrid loss (MSE+MAE)** → Mevcut modeli fine-tune
3. ✅ **b-değer ağırlıklandırma** → Loss function güncelle

### Faz 2: Gerçek Veri Adaptasyonu (3-5 gün)
4. ✅ **Challenge verisiyle fine-tuning** → Yüksek SNR voxeller
5. ✅ **Correlation loss ekleme** → Loss function genişlet

### Faz 3: Gelişmiş Teknikler (1-2 hafta)
6. ✅ **Two-stage training** → Yeni eğitim stratejisi
7. ✅ **Ensemble modeller** → Farklı checkpointları birleştir

---

## 📊 Beklenen İyileşmeler

| Parametre | Mevcut Sorun | Hedef Metrik | Beklenen İyileşme |
|-----------|--------------|--------------|-------------------|
| f Spearman | -0.37 | > 0.5 | Hybrid loss + range fix |
| f MAE | 0.217 | < 0.19 | MAE loss ekleme |
| D* Spearman | -0.63 | > 0.3 | b-ağırlık + fine-tune |
| D* MAE | 23.84 | < 20.0 | Weighted loss + MAE |

---

## 🔧 Kod Dosyaları

Hazırlanan implementasyon dosyaları:
- `train_pia_improved_loss.py` - Hybrid loss ile eğitim
- `finetune_with_challenge_data.py` - Gerçek veriyle fine-tune
- `evaluate_ensemble.py` - Ensemble değerlendirme

Sonraki adım için hangisini oluşturayım?
