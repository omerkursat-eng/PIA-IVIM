"""
PIA Training Script - High Noise Focus
Target: Beat NLLS at σ >= 0.01, achieve linear MAE curve
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from PIA import PIA
from utils import get_batch_optimized

# ----------------------------
# Metrics
# ----------------------------
def mae(x, y):
    return torch.mean(torch.abs(x - y)).item()

def bias(x, y):
    return torch.mean(x - y).item()

def spearman_corr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().flatten().cpu()
    y = y.detach().flatten().cpu()
    if len(x) < 2:
        return 0.0
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    x_rank = (x_rank - x_rank.mean()) / (x_rank.std() + 1e-12)
    y_rank = (y_rank - y_rank.mean()) / (y_rank.std() + 1e-12)
    return torch.mean(x_rank * y_rank).item()

def normalized_mae(mae_val: float, vmin: float, vmax: float) -> float:
    return mae_val / max((vmax - vmin), 1e-12)


# ----------------------------
# Per-Sigma Validation Set Builder
# ----------------------------
@torch.no_grad()
def build_val_set_per_sigma(sigma, n_samples, seed, device):
    """Build validation set for a single sigma value."""
    rng_state_np = np.random.get_state()
    rng_state_torch = torch.random.get_rng_state()
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    noisy, f_gt, dt_gt, ds_gt, clean = get_batch_optimized(
        batch_size=n_samples, noise_sdt=sigma, normalize_b0=True
    )
    
    X = noisy.to(device)
    f = f_gt.to(device)
    dt = dt_gt.to(device)
    ds = ds_gt.to(device)
    
    np.random.set_state(rng_state_np)
    torch.random.set_rng_state(rng_state_torch)
    
    return X, f, dt, ds


# ----------------------------
# Per-Sigma Evaluation
# ----------------------------
@torch.no_grad()
def evaluate_per_sigma(model, X, f_gt, dt_gt, ds_gt):
    """Evaluate model on a single sigma validation set."""
    model.eval()
    f_hat, dt_hat, ds_hat = model.encode(X)
    
    f_hat = f_hat.view(-1)
    dt_hat = dt_hat.view(-1)
    ds_hat = ds_hat.view(-1)
    f_gt = f_gt.view(-1)
    dt_gt = dt_gt.view(-1)
    ds_gt = ds_gt.view(-1)
    
    return {
        "mae_f": mae(f_hat, f_gt),
        "mae_dt": mae(dt_hat, dt_gt),
        "mae_ds": mae(ds_hat, ds_gt),
        "rho_f": spearman_corr_torch(f_hat, f_gt),
        "rho_dt": spearman_corr_torch(dt_hat, dt_gt),
        "rho_ds": spearman_corr_torch(ds_hat, ds_gt),
        "bias_f": bias(f_hat, f_gt),
        "bias_dt": bias(dt_hat, dt_gt),
        "bias_ds": bias(ds_hat, ds_gt),
    }


# ----------------------------
# High-Noise Focused Sigma Sampling
# ----------------------------
def sample_sigma_high_noise_focus(step, warmup_steps, total_steps):
    """
    Curriculum: Start with mid-noise, gradually shift to high-noise focus.
    
    σ ranges:
    - Low:  [1e-3, 5e-3]     - Easy (NLLS works fine here)
    - Mid:  [1e-2, 5e-2]     - Transition zone (target region starts)
    - High: [5e-2, 2e-1]     - Hard (PIA should beat NLLS here)
    """
    
    # Warmup: focus on learnable range
    if step < warmup_steps:
        # Mix of low and mid
        if np.random.rand() < 0.3:
            return np.random.uniform(1e-3, 5e-3)
        else:
            return np.random.uniform(1e-2, 5e-2)
    
    # After warmup: Focused fine-tuning with balanced retention
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    
    # Fine-tuning mode: High-noise focused but keep low-noise alive
    p_low = max(0.05, 0.10 - 0.05 * progress)   # 0.10 -> 0.05 (keep alive to prevent forgetting)
    p_mid = max(0.15, 0.20 - 0.05 * progress)   # 0.20 -> 0.15 (moderate presence)
    p_high = min(0.80, 0.70 + 0.10 * progress)  # 0.70 -> 0.80 (dominant but not exclusive)
    
    # Normalize
    total = p_low + p_mid + p_high
    p_low, p_mid, p_high = p_low/total, p_mid/total, p_high/total
    
    r = np.random.rand()
    if r < p_low:
        return np.random.uniform(1e-3, 5e-3)
    elif r < p_low + p_mid:
        return np.random.uniform(1e-2, 5e-2)
    else:
        # High noise - focused on problem region but cover full range
        if np.random.rand() < 0.75:  # 75% in critical region
            return np.random.uniform(8e-2, 2e-1)  # σ=0.08-0.2 (problem zone)
        else:  # 25% in transition
            return np.random.uniform(5e-2, 8e-2)  # σ=0.05-0.08 (bridge)


# ----------------------------
# Combined Loss: MSE + Param + Correlation
# ----------------------------
def correlation_loss(pred, target):
    """
    Loss that encourages high correlation.
    Returns 1 - correlation (so minimizing this maximizes correlation)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered**2).sum() * (target_centered**2).sum() + 1e-12)
    
    corr = numerator / denominator
    return 1.0 - corr


def param_loss_robust(f_hat, dt_hat, ds_hat, f_gt, dt_gt, ds_gt, use_corr_loss=True):
    """
    Combined loss: SmoothL1 (MAE-like) + Correlation loss
    """
    f_hat = f_hat.view(-1)
    dt_hat = dt_hat.view(-1)
    ds_hat = ds_hat.view(-1)
    f_gt = f_gt.view(-1)
    dt_gt = dt_gt.view(-1)
    ds_gt = ds_gt.view(-1)
    
    # Normalized SmoothL1 (for MAE optimization)
    ef = (f_hat - f_gt) / 1.0
    edt = (dt_hat - dt_gt) / 2.9
    eds = (ds_hat - ds_gt) / 60.0
    
    l_mae_f = F.smooth_l1_loss(ef, torch.zeros_like(ef))
    l_mae_dt = F.smooth_l1_loss(edt, torch.zeros_like(edt))
    l_mae_ds = F.smooth_l1_loss(eds, torch.zeros_like(eds))
    
    # Increase D* weight since it has highest MAE
    loss_mae = 0.30 * l_mae_f + 0.30 * l_mae_dt + 0.40 * l_mae_ds
    
    if use_corr_loss:
        # Correlation loss (for Spearman optimization)
        l_corr_f = correlation_loss(f_hat, f_gt)
        l_corr_dt = correlation_loss(dt_hat, dt_gt)
        l_corr_ds = correlation_loss(ds_hat, ds_gt)
        
        # Same weights for correlation
        loss_corr = 0.30 * l_corr_f + 0.30 * l_corr_dt + 0.40 * l_corr_ds
        
        return loss_mae + 0.5 * loss_corr
    
    return loss_mae


# ----------------------------
# Main Training Loop
# ----------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Convert no_corr_loss to use_corr_loss
    use_corr_loss = not args.no_corr_loss
    
    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"🚀 Training on: {device}")
    
    # Model
    model = PIA(predictor_depth=args.predictor_depth, device=device)
    
    params = (
        list(model.encoder.parameters()) +
        list(model.f_predictor.parameters()) +
        list(model.Dt_predictor.parameters()) +
        list(model.Dstar_predictor.parameters())
    )
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler - Smooth cosine decay for stable high-noise training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.max_steps - args.warmup_steps,  # Cosine decay after warmup
        eta_min=5e-6  # Minimum LR (raised for fine-tuning stability)
    )
    
    # ----------------------------
    # Resume
    # ----------------------------
    start_step = 0
    best_high_noise_score = float("inf")
    best_step = -1
    
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        best_high_noise_score = ckpt.get("best_high_noise_score", float("inf"))
        print(f"✅ Resumed from {args.resume} at step {start_step}")
    
    # ----------------------------
    # Per-Sigma Validation Sets
    # ----------------------------
    # Key sigma values for tracking MAE curve
    val_sigmas = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 1.5e-1, 2e-1]
    
    print("Building per-sigma validation sets...")
    val_sets = {}
    for i, sigma in enumerate(val_sigmas):
        X, f, dt, ds = build_val_set_per_sigma(
            sigma=sigma, 
            n_samples=args.val_samples_per_sigma,
            seed=args.val_seed + i * 100,
            device=device
        )
        val_sets[sigma] = (X, f, dt, ds)
        print(f"  σ={sigma:.1e}: {X.shape[0]} samples")
    
    print(f"\n📊 Tracking {len(val_sigmas)} sigma levels for MAE curve\n")
    
    # ----------------------------
    # Training Loop
    # ----------------------------
    running_loss = 0.0
    t0 = time.time()
    no_improve = 0
    step = start_step  # Initialize step variable
    
    for step in range(start_step + 1, args.max_steps + 1):
        model.train()
        
        # Sample sigma with high-noise focus
        sigma = sample_sigma_high_noise_focus(step, args.warmup_steps, args.max_steps)
        
        # Get batch
        noisy, f_gt, dt_gt, ds_gt, clean = get_batch_optimized(
            batch_size=args.batch_size, noise_sdt=sigma, normalize_b0=True
        )
        noisy = noisy.to(device)
        clean = clean.to(device)
        f_gt = f_gt.to(device)
        dt_gt = dt_gt.to(device)
        ds_gt = ds_gt.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        f_hat, dt_hat, ds_hat = model.encode(noisy)
        recon = model.decode(f_hat, dt_hat, ds_hat)
        
        # Losses
        loss_recon = model.loss_function(recon, clean)
        loss_param = param_loss_robust(
            f_hat, dt_hat, ds_hat, f_gt, dt_gt, ds_gt,
            use_corr_loss=use_corr_loss
        )
        
        loss = loss_recon + args.lam_param * loss_param
        
        # Weighted loss for high-noise batches (Balanced fine-tuning)
        if sigma >= 0.1:
            loss = loss * 1.8  # 1.8x weight for high noise (balanced fine-tuning)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        
        # Print progress
        if step % args.print_every == 0:
            avg_loss = running_loss / args.print_every
            running_loss = 0.0
            elapsed = (time.time() - t0) / 60.0
            lr_now = optimizer.param_groups[0]['lr']
            print(f"[Step {step:06d}] Loss={avg_loss:.5e} | σ={sigma:.2e} | LR={lr_now:.2e} | {elapsed:.1f}m")
        
        # ----------------------------
        # Evaluation & Checkpointing
        # ----------------------------
        if step % args.eval_every == 0:
            print("\n" + "="*100)
            print(f"📊 EVALUATION @ Step {step}")
            print("="*100)
            
            # Evaluate on each sigma
            results = {}
            for val_sigma in val_sigmas:
                X, f, dt, ds = val_sets[val_sigma]
                m = evaluate_per_sigma(model, X, f, dt, ds)
                results[val_sigma] = m
            
            # Print MAE table (key for seeing linear behavior)
            print("\n📈 MAE vs Sigma (Target: Linear growth after σ=0.01)")
            print("-"*80)
            print(f"{'Sigma':>10} | {'MAE_f':>8} | {'MAE_Dt':>8} | {'MAE_D*':>8} | {'ρ_f':>6} | {'ρ_Dt':>6} | {'ρ_D*':>6}")
            print("-"*80)
            
            high_noise_maes = []
            for val_sigma in val_sigmas:
                m = results[val_sigma]
                is_high = val_sigma >= 0.01
                marker = "→" if is_high else " "
                print(f"{marker} {val_sigma:>8.1e} | {m['mae_f']:>8.4f} | {m['mae_dt']:>8.4f} | {m['mae_ds']:>8.4f} | {m['rho_f']:>6.3f} | {m['rho_dt']:>6.3f} | {m['rho_ds']:>6.3f}")
                
                if is_high:
                    # Aggregate high-noise performance
                    total_mae = (
                        normalized_mae(m['mae_f'], 0, 1) +
                        normalized_mae(m['mae_dt'], 0, 2.9) +
                        normalized_mae(m['mae_ds'], 0, 60)
                    ) / 3.0
                    
                    # Give 2x weight to extreme high noise (σ >= 0.1)
                    if val_sigma >= 0.1:
                        high_noise_maes.append(total_mae * 2.0)  # Double penalty for σ≥0.1
                    else:
                        high_noise_maes.append(total_mae)
            
            print("-"*80)
            
            # Calculate high-noise score (weighted toward σ >= 0.1)
            high_noise_score = np.mean(high_noise_maes) if high_noise_maes else float("inf")
            print(f"\n🎯 High-Noise Score (σ≥0.01): {high_noise_score:.6f}")
            
            # Save best checkpoint
            if high_noise_score < best_high_noise_score:
                best_high_noise_score = high_noise_score
                best_step = step
                no_improve = 0
                
                ckpt_path = os.path.join(args.save_dir, "best_high_noise.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_high_noise_score": best_high_noise_score,
                    "results": results,
                    "args": vars(args),
                }, ckpt_path)
                print(f"✅ NEW BEST! Saved to {ckpt_path}")
            else:
                no_improve += 1
                print(f"No improvement ({no_improve}/{args.patience})")
            
            # Save periodic checkpoint
            if step % (args.eval_every * 5) == 0:
                periodic_path = os.path.join(args.save_dir, f"checkpoint_step{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "results": results,
                    "args": vars(args),
                }, periodic_path)
                print(f"💾 Periodic checkpoint: {periodic_path}")
            
            print("="*100 + "\n")
            
            # Early stopping
            if no_improve >= args.patience:
                print("🛑 Early stopping triggered.")
                break
    
    # Final save
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "best_high_noise_score": best_high_noise_score,
        "best_step": best_step,
        "args": vars(args),
    }, final_path)
    
    print(f"\n✅ Training complete!")
    print(f"   Best High-Noise Score: {best_high_noise_score:.6f} @ Step {best_step}")
    print(f"   Final model: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIA High-Noise Training")
    
    # Device & Model
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--predictor_depth", type=int, default=3)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    
    # Loss
    parser.add_argument("--lam_param", type=float, default=1.0)
    parser.add_argument("--no_corr_loss", action="store_true", help="Disable correlation loss")
    
    # Validation
    parser.add_argument("--eval_every", type=int, default=2500)
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--val_samples_per_sigma", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (number of evaluations)")
    parser.add_argument("--val_seed", type=int, default=42)
    
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="pia_runs/high_noise_focus")
    parser.add_argument("--resume", type=str, default="")
    
    args = parser.parse_args()
    main(args)
