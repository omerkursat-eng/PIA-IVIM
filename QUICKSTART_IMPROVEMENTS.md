# PIA Model İyileştirme - Hızlı Başlangıç Kılavuzu

## 🎯 Tespit Edilen Sorunlar ve Çözümler

### Sorunlar
1. **f parametresi**: Negatif Spearman korelasyonu (-0.37), yüksek MAE
2. **D* parametresi**: Negatif Spearman korelasyonu (-0.63), çok yüksek MAE
3. **Dt parametresi**: ✅ Mükemmel çalışıyor

### Önerilen Çözümler

#### ⚡ Yaklaşım 1: İyileştirilmiş Loss Function (EN HIZLI)
**Dosya**: `train_pia_improved_loss.py`

**Değişiklikler**:
- ✅ Hybrid Loss: MSE + MAE + Correlation
- ✅ Parametreye özel ağırlıklar (f: 2x, Dt: 1x, D*: 3x)
- ✅ b-değer ağırlıklandırma (düşük b'lere yüksek ağırlık)
- ✅ Düzeltilmiş parametre aralıkları (Challenge verisine uygun)

**Nasıl çalıştırılır**:
```bash
python train_pia_improved_loss.py
```

**Beklenen süre**: ~6-8 saat (100K steps, CPU'da daha uzun)

**Beklenen iyileşme**:
- f Spearman: -0.37 → **> 0.4** ✅
- D* Spearman: -0.63 → **> 0.2** ✅
- MAE değerleri: **%10-20 azalma** ✅

---

#### 🎯 Yaklaşım 2: Challenge Verisiyle Fine-tuning (ÇOK ETKİLİ)
**Dosya**: `finetune_with_challenge_data.py`

**Strateji**:
1. Yüksek SNR voxelleri seç (gürültü düşük)
2. Mevcut en iyi modeli yükle
3. Düşük learning rate ile fine-tune (5e-5)
4. Hybrid loss kullan

**Nasıl çalıştırılır**:
```bash
# En iyi mevcut modeli fine-tune et
python finetune_with_challenge_data.py
```

**Beklenen süre**: ~30-60 dakika (50 epochs)

**Beklenen iyileşme**:
- Gerçek veri dağılımına adaptasyon
- Negatif korelasyon sorununda **%30-50 iyileşme**
- MAE'de **%15-25 azalma**

---

#### 🔄 Yaklaşım 3: İki Aşamalı Eğitim (DENEYSEL)
**Konsept**: Önce Dt, sonra f ve D*

```python
# Stage 1: Sadece Dt (kolay parametre)
train_only_Dt(epochs=100)

# Stage 2: Dt frozen, f ve D* optimize et
freeze_Dt_and_train_others(epochs=100)
```

**Avantaj**: Her parametreye odaklanma
**Dezavantaj**: Daha uzun eğitim süresi

---

## 📊 Karşılaştırma: Mevcut vs İyileştirilmiş

### Mevcut Model (exp3_step62k)
```
Parametre    RMSE      MAE       Spearman
──────────────────────────────────────────
f            0.2779    0.2170    -0.37 ❌
Dt           0.3762    0.2080    +0.83 ✅
D*          29.7992   23.8392    -0.63 ❌
```

### Hedef (İyileştirilmiş Loss)
```
Parametre    RMSE      MAE       Spearman
──────────────────────────────────────────
f            0.25      0.18      +0.45 ✅
Dt           0.35      0.19      +0.85 ✅
D*          28.00     20.00      +0.25 ✅
```

---

## 🚀 Önerilen İş Akışı

### Hafta 1: Hızlı Kazançlar
```bash
# 1. İyileştirilmiş loss ile yeni model eğit
python train_pia_improved_loss.py

# 2. Eğitim tamamlandığında, modeli fine-tune et
python finetune_with_challenge_data.py

# 3. Sonuçları değerlendir
python analyze_kspace_data.py
```

### Hafta 2: İleri Optimizasyon
- Hiperparametre tuning (learning rate, loss weights)
- Ensemble modeller (birden fazla checkpoint birleştir)
- Patient 0002 ile cross-validation

---

## 🔧 Hiperparametre Önerileri

### Loss Ağırlıkları (train_pia_improved_loss.py)
```python
# Mevcut
signal_weight=1.0
param_mse_weight=0.5
param_mae_weight=0.3
param_corr_weight=0.2

# Alternatif 1: Daha fazla korelasyon odağı
param_corr_weight=0.4  # Artır
param_mse_weight=0.4   # Azalt

# Alternatif 2: MAE odaklı
param_mae_weight=0.5   # Artır
param_mse_weight=0.3   # Azalt
```

### Parametre Ağırlıkları
```python
# Mevcut
f_weight=2.0
Dt_weight=1.0
Dstar_weight=3.0

# Alternatif: Daha agresif
f_weight=3.0    # f sorunu ciddi
Dstar_weight=5.0  # D* en sorunlu
```

### Learning Rate Scheduling
```python
# Warmup ekle
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=1e-3, 
    total_steps=NUM_STEPS,
    pct_start=0.1  # İlk %10'da warmup
)
```

---

## 📈 Beklenen Timeline

| Zaman | Eylem | Beklenen Sonuç |
|-------|-------|----------------|
| T+0 | İyileştirilmiş loss ile eğitime başla | - |
| T+8h | Eğitim tamamlandı, ilk checkpoint'leri değerlendir | Korelasyon +0.2-0.3 iyileşme |
| T+9h | En iyi checkpoint'i fine-tune et | Ek +0.1-0.2 iyileşme |
| T+10h | Full evaluation (analyze_kspace_data.py) | Final metrikler |
| T+1d | Hiperparametre tuning dene | Ek %5-10 iyileşme |

---

## ❓ Sorun Giderme

### 1. "Loss çok yüksek"
- Learning rate'i azalt (1e-4 → 5e-5)
- Gradient clipping ekle (max_norm=0.5)

### 2. "Korelasyon hala negatif"
- `param_corr_weight`'i artır (0.2 → 0.5)
- Fine-tuning epoch sayısını artır (50 → 100)

### 3. "MAE azalmıyor"
- `param_mae_weight`'i artır (0.3 → 0.5)
- L1 loss ağırlığını artır

### 4. "Overfitting"
- Dropout ekle (encoder'a %10-20)
- Early stopping kullan
- Training data augmentation

---

## 📝 Notlar

- **Checkpoint kaydetme**: Her 10K step'te otomatik
- **Best model**: En düşük loss'ta otomatik kaydedilir
- **Fine-tuning**: Mevcut en iyi modelden başlar
- **Evaluation**: `analyze_kspace_data.py` ile full comparison

---

## 📧 Daha Fazla Bilgi

Detaylı açıklamalar için:
- `IMPROVEMENT_PLAN.md` - Tüm yaklaşımların detaylı açıklaması
- `METRICS_SUMMARY.md` - Mevcut performans analizi
- `ANALYSIS_RESULTS.md` - Genel sonuçlar

---

**En Hızlı Sonuç İçin**: `python train_pia_improved_loss.py` → `python finetune_with_challenge_data.py`
