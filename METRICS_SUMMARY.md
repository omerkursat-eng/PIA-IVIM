# IVIM Parametre Tahmini - Detaylı Metrik Karşılaştırması

## 📊 Tüm Modellerin Karşılaştırmalı Metrikleri

### f (Perfusion Fraction) Parametresi

| Model | RMSE ↓ | NRMSE ↓ | MAE ↓ | Spearman ↑ | RMSE İyileşme |
|-------|--------|---------|-------|------------|---------------|
| **NLLS** | 0.3253 | 1.04 | **0.1791** | **0.29** | baseline |
| exp2_best | **0.2754** | **0.88** | 0.2205 | -0.33 | +15.3% ✅ |
| exp2_final | 0.2810 | 0.90 | 0.2237 | -0.34 | +13.6% ✅ |
| exp3_50k | 0.2791 | 0.89 | 0.2213 | -0.34 | +14.2% ✅ |
| exp3_62k ⭐ | 0.2779 | 0.89 | **0.2170** | -0.37 | +14.6% ✅ |

**Analiz**:
- ✅ Tüm PIA modelleri RMSE'de NLLS'den %13-15 daha iyi
- ⚠️ MAE değerleri NLLS'den biraz yüksek
- ❌ Spearman korelasyonu negatif (sistematik bias var)

---

### Dt (Tissue Diffusion) Parametresi

| Model | RMSE ↓ | NRMSE ↓ | MAE ↓ | Spearman ↑ | RMSE İyileşme |
|-------|--------|---------|-------|------------|---------------|
| **NLLS** | 0.5405 | 0.26 | 0.2312 | 0.64 | baseline |
| exp2_best | 0.3892 | 0.19 | 0.2173 | **0.82** | +28.0% ✅ |
| exp2_final | 0.3998 | 0.19 | 0.2249 | 0.82 | +26.0% ✅ |
| exp3_50k | 0.4051 | 0.19 | 0.2246 | 0.82 | +25.1% ✅ |
| exp3_62k ⭐ | **0.3762** | **0.18** | **0.2080** | **0.83** | +30.4% ✅✅✅ |

**Analiz**:
- ✅✅✅ **En iyi performans!** PIA modelleri RMSE'de %25-30 daha iyi
- ✅ MAE'de de NLLS'den daha iyi
- ✅ Spearman korelasyonu 0.64'ten 0.83'e çıktı (güçlü korelasyon)
- ⭐ **exp3_62k tüm metriklerde kazanıyor**

---

### D* (Pseudo-diffusion) Parametresi

| Model | RMSE ↓ | NRMSE ↓ | MAE ↓ | Spearman ↑ | RMSE İyileşme |
|-------|--------|---------|-------|------------|---------------|
| **NLLS** | **29.31** | **0.50** | **18.60** | **0.21** | baseline |
| exp2_best | 30.01 | 0.51 | 23.88 | -0.66 | -2.4% ❌ |
| exp2_final | 30.44 | 0.52 | 24.08 | -0.66 | -3.9% ❌ |
| exp3_50k | 30.35 | 0.51 | 24.60 | -0.69 | -3.6% ❌ |
| exp3_62k ⭐ | 29.80 | 0.50 | 23.84 | -0.63 | -1.7% ≈ |

**Analiz**:
- ≈ PIA ve NLLS RMSE açısından benzer
- ❌ PIA MAE değerleri daha yüksek
- ❌ Spearman korelasyonu negatif (sistematik bias var)
- ⚠️ D* tahmini zor bir parametre - her iki yöntem de zorlanıyor

---

## 🏆 Genel Değerlendirme

### En İyi Model: **exp3_step62k**

| Metrik | NLLS | PIA (exp3_62k) | Kazanan |
|--------|------|----------------|---------|
| **f RMSE** | 0.3253 | 0.2779 (-14.6%) | PIA ✅ |
| **f MAE** | 0.1791 | 0.2170 (+21.2%) | NLLS ✅ |
| **f Spearman** | 0.29 | -0.37 | NLLS ✅ |
| **Dt RMSE** | 0.5405 | 0.3762 (-30.4%) | PIA ✅✅✅ |
| **Dt MAE** | 0.2312 | 0.2080 (-10.0%) | PIA ✅ |
| **Dt Spearman** | 0.64 | 0.83 (+29.7%) | PIA ✅✅ |
| **D* RMSE** | 29.31 | 29.80 (+1.7%) | NLLS ✅ |
| **D* MAE** | 18.60 | 23.84 (+28.2%) | NLLS ✅ |
| **D* Spearman** | 0.21 | -0.63 | NLLS ✅ |

### Skor Tablosu
- **PIA Kazanma**: 5 metrik (özellikle Dt'de dominant)
- **NLLS Kazanma**: 4 metrik (özellikle MAE ve D* tahmininde)

---

## 💡 Öneriler

### PIA Modeli İçin İyileştirmeler:

1. **f ve D* parametrelerindeki negatif korelasyon sorunu**:
   - Model bu parametrelerde sistematik bias yapıyor
   - Çözüm: Loss function'a korelasyon terimi eklenebilir
   - Alternatif: Daha fazla çeşitli veriyle eğitim

2. **MAE optimizasyonu**:
   - Şu an MSE loss kullanılıyor
   - Hybrid loss (MSE + MAE) denenmeli

3. **D* tahmini için özel stratejiler**:
   - D* yüksek değişkenlik gösteren bir parametre
   - Düşük b-değerlerine daha fazla ağırlık verilmeli
   - Segmented fitting yaklaşımı (önce Dt, sonra f ve D*)

### NLLS için Optimizasyonlar:

1. **Başlangıç değerleri iyileştirmesi**:
   - PIA modelinden başlangıç değerleri alınabilir
   - İki aşamalı hibrit yaklaşım: PIA → NLLS refinement

2. **Gürültü seviyesine göre adaptive fitting**:
   - Düşük SNR bölgelerde constraints sıkılaştırılmalı

---

## 📈 Sonuç

**Dt parametresi tahmininde PIA modeli açık ara kazanıyor** (+30% RMSE, +30% Spearman)

**f parametresi için RMSE iyi ama korelasyon sorunlu** - iyileştirme gerekli

**D* parametresi her iki yöntem için de challenging** - benzer performans

**Genel öneri**: 
- Dt tahmini için: **PIA model (exp3_step62k)** kullan ⭐
- f ve D* için: Hibrit yaklaşım (PIA + NLLS refinement) denenebilir
