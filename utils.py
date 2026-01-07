import numpy as np
from scipy.optimize import curve_fit
import torch


def read_data(file_dir, fname, i):
    
    fname_tmp = file_dir + "{:04}".format(i) + fname
    data = np.load(fname_tmp)
    
    return data


def rRMSE(x,y,t, is_f=False):
    
    Nx, Ny = x.shape

    t_tmp = np.reshape(t, (Nx*Ny,))
    tumor_indice = np.argwhere(t_tmp == 8)
    non_tumor_indice = np.argwhere(t_tmp != 8)
    non_air_indice = np.argwhere(t_tmp != 1)
    non_tumor_air_indice= np.intersect1d(non_tumor_indice,non_air_indice)
    
    x_tmp = np.reshape(x, (Nx*Ny,))
    x_t = x_tmp[tumor_indice]
    x_nt = x_tmp[non_tumor_air_indice]
    
    y_tmp = np.reshape(y, (Nx*Ny,))
    y_t = y_tmp[tumor_indice]
    y_nt = y_tmp[non_tumor_air_indice]
    
    # tumor region
    tmp1 = np.sqrt(tumor_indice.shape[0]) if is_f else np.sqrt(np.sum(np.square(y_t)))
    tmp2 = np.sqrt(np.sum(np.square(x_t-y_t)))
    z_t = tmp2 / tmp1
    
    # non-tumor region
    tmp1 = np.sqrt(non_tumor_air_indice.shape[0]) if is_f else np.sqrt(np.sum(np.square(y_nt)))
    tmp2 = np.sqrt(np.sum(np.square(x_nt-y_nt)))
    z_nt = tmp2 / tmp1
    
    return z_t, z_nt

def rRMSE_per_case(x_f,x_dt,x_ds,y_f,y_dt,y_ds,t):
    
    
    R_f_t, R_f_nt = rRMSE(x_f, y_f, t, is_f=True)
    R_Dt_t, R_Dt_nt = rRMSE(x_dt, y_dt, t)
    R_Ds_t, R_Ds_nt = rRMSE(x_ds, y_ds, t)
    
    z =  (R_f_t + R_Dt_t + R_Ds_t)/3 + (R_f_nt + R_Dt_nt)/2
    
    z_t =  (R_f_t + R_Dt_t + R_Ds_t)/3
    
    return z, z_t


def rRMSE_all_cases(x_f,x_dt,x_ds,y_f,y_dt,y_ds,t):
    
    z = np.empty([x_f.shape[2]])
    z_t = np.empty([x_f.shape[2]])
    
    for i in range(x_f.shape[2]):
        z[i], z_t[i] = rRMSE_per_case(x_f[:,:,i],x_dt[:,:,i],x_ds[:,:,i],y_f[:,:,i],y_dt[:,:,i],y_ds[:,:,i],t[:,:,i]) 
        
    return np.average(z), np.average(z_t)


def ivim_fit_func(b, f, Dt, Dstar):
    """IVIM bi-exponential model for curve fitting."""
    return (1 - f) * np.exp(-b/1000 * Dt) + f * np.exp(-b/1000 * Dstar)


def get_batch_optimized(batch_size=16, noise_sdt=0.01, normalize_b0=True):
    # 1. b-values (s/mm^2)
    b_values = np.array([0, 5, 50, 100, 200, 500, 800, 1000], dtype=np.float32)
    
    # 2. Parametreleri Fiziksel Sınırlara Göre Örnekle
    f = np.random.uniform(0.0, 1.0, size=(batch_size, 1)).astype(np.float32)
    
    # Dt (Tissue Diffusion): Yavaş difüzyon (Genelde < 3.0 e-3)
    Dt = np.random.uniform(0.0, 2.9, size=(batch_size, 1)).astype(np.float32)
    
    # D_star (Pseudo-diffusion): Hızlı akış. 
    # PIA model aralığı [0, 60] ile uyumlu
    D_star = np.random.uniform(0.0, 60.0, size=(batch_size, 1)).astype(np.float32)



    # 3. Temiz Sinyal Üretimi (Vektörize)
    # b değerlerini modelin birimine uygun hale getir (x10^-3 mm^2/s varsayımıyla /1000 yapıyoruz)
    b_scaled = (b_values[None, :] / 1000.0).astype(np.float32)
    
    # IVIM Modeli: S/S0 = (1-f)*exp(-b*Dt) + f*exp(-b*D*)
    # Not: S0 = 1 kabul ediyoruz çünkü aşağıda zaten normalize edeceğiz veya 1 üreteceğiz.
    clean_signal = (1.0 - f) * np.exp(-b_scaled * Dt) + f * np.exp(-b_scaled * D_star)

    # 4. Rician Noise Ekleme
    noise_real = np.random.normal(0.0, noise_sdt, size=clean_signal.shape).astype(np.float32)
    noise_imag = np.random.normal(0.0, noise_sdt, size=clean_signal.shape).astype(np.float32)
    
    # Rician Magnitude: sqrt((S + nr)^2 + ni^2)
    noisy_signal = np.sqrt((clean_signal + noise_real) ** 2 + noise_imag ** 2).astype(np.float32)


    # 5. Normalizasyon (Opsiyonel ama önerilir)
    if normalize_b0:
        # Her örneğin kendi gürültülü b=0 değerine bölüyoruz
        b0_val = noisy_signal[:, [0]] # Shape (batch, 1) keeps dimension
        # 0'a bölme hatasını engellemek için clip
        noisy_signal = noisy_signal / np.clip(b0_val, 1e-8, None)

    return (
        torch.from_numpy(noisy_signal).float(), # Input to Neural Net
        torch.from_numpy(f).float(),            # Ground Truth f
        torch.from_numpy(Dt).float(),           # Ground Truth Dt
        torch.from_numpy(D_star).float(),       # Ground Truth D_star
        torch.from_numpy(clean_signal).float()  # Ground Truth Signal (for debugging/loss)
    )


def hybrid_fit(signals, bvals=[0, 5, 50, 100, 200, 500, 800, 1000]):
    """NLLS fitting for IVIM parameters."""
    numcols, acquisitions = signals.shape
    f = np.zeros((numcols,))
    Dt = np.zeros((numcols,))
    Dstar = np.zeros((numcols,))
    for col in range(numcols):
        voxel = signals[col]
        xdata = np.array(bvals)
        ydata = voxel.ravel()
        try:
            fitdata_, _ = curve_fit(
                ivim_fit_func,
                xdata,
                ydata,
                p0=[0.15, 1.5, 8],
                bounds=([0, 0, 0], [1, 2.9, 60]),  # D* alt sınırı 0 (PIA ile uyumlu)
                method='trf',
                maxfev=5000
            )
        except RuntimeError:
            fitdata_ = [0.15, 1.5, 8]
            
        coeffs = fitdata_
        f[col] = coeffs[0]
        Dt[col] = coeffs[1]
        Dstar[col] = coeffs[2]
    return f, Dt, Dstar




  




