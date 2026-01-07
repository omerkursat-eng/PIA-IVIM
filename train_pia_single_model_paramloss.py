import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from PIA import PIA
from utils import get_batch_optimized


# ----------------------------
# Helpers: metrics
# ----------------------------
def mae(x, y):
    return torch.mean(torch.abs(x - y)).item()

def bias(x, y):
    return torch.mean(x - y).item()

def spearman_corr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().flatten().cpu()
    y = y.detach().flatten().cpu()
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    x_rank = (x_rank - x_rank.mean()) / (x_rank.std() + 1e-12)
    y_rank = (y_rank - y_rank.mean()) / (y_rank.std() + 1e-12)
    return torch.mean(x_rank * y_rank).item()

def normalized_mae(mae_val: float, vmin: float, vmax: float) -> float:
    return mae_val / max((vmax - vmin), 1e-12)

def highscore_from_maes(mae_f, mae_dt, mae_ds):
    nf  = normalized_mae(mae_f,  0.0, 1.0)
    ndt = normalized_mae(mae_dt, 0.0, 2.9)
    nds = normalized_mae(mae_ds, 0.0, 60.0)
    return (nf + ndt + nds) / 3.0

def outlier_rate(abs_err: torch.Tensor, vmin: float, vmax: float, frac: float = 0.10) -> float:
    thr = frac * (vmax - vmin)
    return torch.mean((abs_err > thr).float()).item()


# ----------------------------
# Fixed validation set builder
# ----------------------------
@torch.no_grad()
def build_val_set(sigmas, n_per_sigma, batch_chunk, seed, device):
    rng_state_np = np.random.get_state()
    rng_state_torch = torch.random.get_rng_state()

    np.random.seed(seed)
    torch.manual_seed(seed)

    xs, fgs, dts, dss = [], [], [], []

    for s in sigmas:
        remaining = n_per_sigma
        while remaining > 0:
            bsz = min(batch_chunk, remaining)
            noisy, f_gt, dt_gt, ds_gt, _clean = get_batch_new(batch_size=bsz, noise_sdt=s, normalize_b0=False)
            xs.append(noisy)
            fgs.append(f_gt)
            dts.append(dt_gt)
            dss.append(ds_gt)
            remaining -= bsz

    X = torch.cat(xs, dim=0).to(device)
    f = torch.cat(fgs, dim=0).to(device)
    dt = torch.cat(dts, dim=0).to(device)
    ds = torch.cat(dss, dim=0).to(device)

    np.random.set_state(rng_state_np)
    torch.random.set_rng_state(rng_state_torch)

    return X, f, dt, ds


# ----------------------------
# Validation evaluator (parameter-based)
# ----------------------------
@torch.no_grad()
def evaluate_params(model, X, f_gt, dt_gt, ds_gt):
    model.eval()
    f_hat, dt_hat, ds_hat = model.encode(X)

    f_hat = f_hat.view(-1)
    dt_hat = dt_hat.view(-1)
    ds_hat = ds_hat.view(-1)

    f_gt = f_gt.view(-1)
    dt_gt = dt_gt.view(-1)
    ds_gt = ds_gt.view(-1)

    mae_f  = mae(f_hat, f_gt)
    mae_dt = mae(dt_hat, dt_gt)
    mae_ds = mae(ds_hat, ds_gt)

    rho_f  = spearman_corr_torch(f_hat, f_gt)
    rho_dt = spearman_corr_torch(dt_hat, dt_gt)
    rho_ds = spearman_corr_torch(ds_hat, ds_gt)

    bias_f  = bias(f_hat, f_gt)
    bias_dt = bias(dt_hat, dt_gt)
    bias_ds = bias(ds_hat, ds_gt)

    of  = outlier_rate(torch.abs(f_hat  - f_gt),  0.0, 1.0, frac=0.10)
    odt = outlier_rate(torch.abs(dt_hat - dt_gt), 0.0, 2.9, frac=0.10)
    ods = outlier_rate(torch.abs(ds_hat - ds_gt), 0.0, 60.0, frac=0.10)

    score = highscore_from_maes(mae_f, mae_dt, mae_ds)

    return {
        "mae_f": mae_f, "mae_dt": mae_dt, "mae_ds": mae_ds,
        "rho_f": rho_f, "rho_dt": rho_dt, "rho_ds": rho_ds,
        "bias_f": bias_f, "bias_dt": bias_dt, "bias_ds": bias_ds,
        "out_f": of, "out_dt": odt, "out_ds": ods,
        "highscore": score
    }


# ----------------------------
# Sigma sampling schedule (3-regime)
# ----------------------------
def sample_sigma_3regime(step, low_sigmas, mid_sigmas, extreme_sigmas,
                         warmup_steps, stage1_steps,
                         p_low_stage1, p_mid_stage1, p_ext_stage1,
                         p_low_stage2, p_mid_stage2, p_ext_stage2,
                         extreme_weights=None):

    if step < warmup_steps:
        return float(np.random.choice(low_sigmas))

    stage1_end = warmup_steps + stage1_steps
    if step < stage1_end:
        p_low, p_mid, p_ext = p_low_stage1, p_mid_stage1, p_ext_stage1
    else:
        p_low, p_mid, p_ext = p_low_stage2, p_mid_stage2, p_ext_stage2

    # safety normalize (in case user passes slightly off sums)
    s = (p_low + p_mid + p_ext)
    if s <= 0:
        p_low, p_mid, p_ext = 0.1, 0.6, 0.3
        s = 1.0
    p_low, p_mid, p_ext = p_low / s, p_mid / s, p_ext / s

    r = np.random.rand()
    if r < p_low:
        return float(np.random.choice(low_sigmas))
    elif r < (p_low + p_mid):
        return float(np.random.choice(mid_sigmas))
    else:
        if (extreme_weights is not None) and (len(extreme_weights) == len(extreme_sigmas)):
            w = np.array(extreme_weights, dtype=np.float64)
            w = w / max(w.sum(), 1e-12)
            return float(np.random.choice(extreme_sigmas, p=w))
        return float(np.random.choice(extreme_sigmas))


# ----------------------------
# PARAM LOSS (normalized, robust)
# ----------------------------
def param_loss_normalized(f_hat, dt_hat, ds_hat, f_gt, dt_gt, ds_gt):
    """
    Robust L1-like loss using SmoothL1 on normalized errors.
    Normalize by PIA ranges: f/1, Dt/2.9, D*/60.
    """
    f_hat = f_hat.view(-1)
    dt_hat = dt_hat.view(-1)
    ds_hat = ds_hat.view(-1)

    f_gt = f_gt.view(-1)
    dt_gt = dt_gt.view(-1)
    ds_gt = ds_gt.view(-1)

    ef  = (f_hat  - f_gt)  / 1.0
    edt = (dt_hat - dt_gt) / 2.9
    eds = (ds_hat - ds_gt) / 60.0

    lf  = F.smooth_l1_loss(ef,  torch.zeros_like(ef))
    ldt = F.smooth_l1_loss(edt, torch.zeros_like(edt))
    lds = F.smooth_l1_loss(eds, torch.zeros_like(eds))

    # weights: emphasize f
    w_f, w_dt, w_ds = 0.35, 0.30, 0.35
    return w_f * lf + w_dt * ldt + w_ds * lds


def lambda_param(step, warmup_steps, ramp_steps, lam_max):
    """
    0 during warmup, then linearly ramps to lam_max over ramp_steps.
    """
    if step < warmup_steps:
        return 0.0
    t = min(max(step - warmup_steps, 0), ramp_steps)
    return lam_max * (t / max(ramp_steps, 1))


# ----------------------------
# Main training
# ----------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    model = PIA(predictor_depth=args.predictor_depth, device=device).to(device)

    params = (
        list(model.encoder.parameters()) +
        list(model.f_predictor.parameters()) +
        list(model.Dt_predictor.parameters()) +
        list(model.Dstar_predictor.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ----------------------------
    # Resume (optional)
    # ----------------------------
    start_step = 0
    best_total = float("inf")
    best_mid   = float("inf")
    best_ext   = float("inf")
    best_low   = float("inf")
    best_step  = -1
    no_improve_evals = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_step = int(ckpt.get("step", start_step))
        best_total = float(ckpt.get("best_total", ckpt.get("best_highscore", best_total)))
        best_mid   = float(ckpt.get("best_mid", best_mid))
        best_ext   = float(ckpt.get("best_ext", best_ext))
        best_low   = float(ckpt.get("best_lowscore", best_low))
        best_step  = int(ckpt.get("best_step", best_step))
        no_improve_evals = 0

        # IMPORTANT: override LR with current args.lr
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr

        print(f"✅ Resumed from: {args.resume}")
        print(f"   start_step={start_step} | best_total={best_total:.6f} | best_mid={best_mid:.6f} | best_ext={best_ext:.6f} | best_low={best_low:.6f}")
        print(f"   LR is set to {optimizer.param_groups[0]['lr']:.3e} (overridden by args.lr)\n")
    else:
        print("🆕 Starting from scratch (no resume).\n")

    # ----------------------------
    # Sigma sets (3-regime)
    # ----------------------------
    sigmas = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3,
              1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2,
              1e-1, 3e-1, 5e-1]

    low_sigmas = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    mid_sigmas = [1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 1e-1]
    extreme_sigmas = [3e-1, 5e-1]

    # ----------------------------
    # Fixed validation (LOW / MID / EXT)
    # ----------------------------
    print("Building fixed validation sets...")
    X_mid, f_mid, dt_mid, ds_mid = build_val_set(
        sigmas=mid_sigmas, n_per_sigma=args.val_mid_per_sigma,
        batch_chunk=args.val_chunk, seed=args.val_seed, device=device
    )
    X_ext, f_ext, dt_ext, ds_ext = build_val_set(
        sigmas=extreme_sigmas, n_per_sigma=args.val_ext_per_sigma,
        batch_chunk=args.val_chunk, seed=args.val_seed + 77, device=device
    )
    X_low, f_low, dt_low, ds_low = build_val_set(
        sigmas=low_sigmas, n_per_sigma=args.val_low_per_sigma,
        batch_chunk=args.val_chunk, seed=args.val_seed + 123, device=device
    )
    print(f"Val-Mid size: {X_mid.shape[0]} | Val-Ext size: {X_ext.shape[0]} | Val-Low size: {X_low.shape[0]}\n")
    print("Training started.\n")

    running_loss = 0.0
    t0 = time.time()

    for step in range(start_step + 1, args.max_steps + 1):
        model.train()

        sigma = sample_sigma_3regime(
            step=step,
            low_sigmas=low_sigmas,
            mid_sigmas=mid_sigmas,
            extreme_sigmas=extreme_sigmas,
            warmup_steps=args.warmup_steps,
            stage1_steps=args.stage1_steps,
            p_low_stage1=args.p_low_stage1,
            p_mid_stage1=args.p_mid_stage1,
            p_ext_stage1=args.p_ext_stage1,
            p_low_stage2=args.p_low_stage2,
            p_mid_stage2=args.p_mid_stage2,
            p_ext_stage2=args.p_ext_stage2,
            extreme_weights=args.extreme_weights
        )

        noisy, f_gt, dt_gt, ds_gt, clean = get_batch_new(batch_size=args.batch_size, noise_sdt=sigma, normalize_b0=False)
        noisy = noisy.to(device)
        clean = clean.to(device)
        f_gt  = f_gt.to(device)
        dt_gt = dt_gt.to(device)
        ds_gt = ds_gt.to(device)

        optimizer.zero_grad()

        f_hat, dt_hat, ds_hat = model.encode(noisy)
        recon = model.decode(f_hat, dt_hat, ds_hat)

        loss_sig = model.loss_function(recon, clean)

        lam_p = lambda_param(step, args.warmup_steps, args.ramp_steps, args.lam_param_max)
        loss_p = param_loss_normalized(f_hat, dt_hat, ds_hat, f_gt, dt_gt, ds_gt)

        loss = (1.0 - lam_p) * loss_sig + lam_p * loss_p
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        running_loss += loss.item()

        if step % args.print_every == 0:
            avg = running_loss / args.print_every
            running_loss = 0.0
            elapsed = (time.time() - t0) / 60.0
            print(f"[Step {step:06d}] train_loss(avg)={avg:.6e} | lam_p={lam_p:.3f} | sigma={sigma:.1e} | elapsed={elapsed:.1f} min")

        if step % args.eval_every == 0:
            m_mid = evaluate_params(model, X_mid, f_mid, dt_mid, ds_mid)
            m_ext = evaluate_params(model, X_ext, f_ext, dt_ext, ds_ext)
            m_low = evaluate_params(model, X_low, f_low, dt_low, ds_low)

            midscore = m_mid["highscore"]
            extscore = m_ext["highscore"]
            lowscore = m_low["highscore"]

            total = args.w_mid * midscore + args.w_ext * extscore

            # track best lowscore (optional health check)
            if lowscore < best_low:
                best_low = lowscore

            low_ok = (lowscore <= best_low * (1.0 + args.low_tolerance)) if args.enforce_low else True

            # extreme safety: allow first time / if best_ext inf
            if np.isinf(best_ext):
                ext_ok = True
            else:
                ext_ok = (extscore <= best_ext * (1.0 + args.ext_tolerance))

            print("\n" + "-"*100)
            print(f"[VAL @ Step {step:06d}] MidScore={midscore:.6f} | ExtScore={extscore:.6f} | Total={total:.6f} | ext_ok={ext_ok} | low_ok={low_ok}")
            print(f"  MID MAE : f={m_mid['mae_f']:.6f} Dt={m_mid['mae_dt']:.6f} D*={m_mid['mae_ds']:.6f} | ρ: f={m_mid['rho_f']:.4f} Dt={m_mid['rho_dt']:.4f} D*={m_mid['rho_ds']:.4f}")
            print(f"  EXT MAE : f={m_ext['mae_f']:.6f} Dt={m_ext['mae_dt']:.6f} D*={m_ext['mae_ds']:.6f} | ρ: f={m_ext['rho_f']:.4f} Dt={m_ext['rho_dt']:.4f} D*={m_ext['rho_ds']:.4f}")
            print(f"  LOW MAE : f={m_low['mae_f']:.6f} Dt={m_low['mae_dt']:.6f} D*={m_low['mae_ds']:.6f}")
            print("-"*100 + "\n")

            # update best_mid/best_ext for tracking
            if midscore < best_mid:
                best_mid = midscore
            if extscore < best_ext:
                best_ext = extscore

            improved = (total < best_total) and ext_ok and low_ok
            if improved:
                best_total = total
                best_step = step
                no_improve_evals = 0

                ckpt_path = os.path.join(args.save_dir, "best_checkpoint.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_total": best_total,
                    "best_mid": best_mid,
                    "best_ext": best_ext,
                    "best_lowscore": best_low,
                    "best_step": best_step,
                    "metrics_mid": m_mid,
                    "metrics_ext": m_ext,
                    "metrics_low": m_low,
                    "args": vars(args),
                }, ckpt_path)
                print(f"✅ Saved BEST checkpoint at step {step} (Total={best_total:.6f}) -> {ckpt_path}\n")
            else:
                no_improve_evals += 1
                print(f"no_improve_evals={no_improve_evals}/{args.patience_evals}\n")

            if no_improve_evals >= args.patience_evals:
                print("🛑 Early stopping triggered.")
                break

    last_path = os.path.join(args.save_dir, "last_checkpoint.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_total": best_total,
        "best_mid": best_mid,
        "best_ext": best_ext,
        "best_lowscore": best_low,
        "best_step": best_step,
        "args": vars(args),
    }, last_path)

    print(f"\n✅ Training finished. Best step: {best_step}, Best Total: {best_total:.6f}")
    print(f"Saved last checkpoint -> {last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIA single-model training (3-regime sigma, mid+ext focus)")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--predictor_depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--print_every", type=int, default=1000)

    # Warmup + stage timing
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--stage1_steps", type=int, default=30000)

    # 3-regime probabilities (Stage1)
    parser.add_argument("--p_low_stage1", type=float, default=0.10)
    parser.add_argument("--p_mid_stage1", type=float, default=0.60)
    parser.add_argument("--p_ext_stage1", type=float, default=0.30)

    # 3-regime probabilities (Stage2+)
    parser.add_argument("--p_low_stage2", type=float, default=0.05)
    parser.add_argument("--p_mid_stage2", type=float, default=0.55)
    parser.add_argument("--p_ext_stage2", type=float, default=0.40)

    # Optional: weights inside extreme pool (for [1e-1, 3e-1, 5e-1])
    # Example: --extreme_weights 0.60 0.25 0.15
    parser.add_argument("--extreme_weights", type=float, nargs="*", default=None)

    # Param-loss ramp
    parser.add_argument("--ramp_steps", type=int, default=20000)
    parser.add_argument("--lam_param_max", type=float, default=0.50)

    # Validation set sizes (per sigma)
    parser.add_argument("--val_mid_per_sigma", type=int, default=2000)
    parser.add_argument("--val_ext_per_sigma", type=int, default=1000)
    parser.add_argument("--val_low_per_sigma", type=int, default=500)
    parser.add_argument("--val_chunk", type=int, default=512)
    parser.add_argument("--val_seed", type=int, default=1234)

    # Checkpoint selection (mid slightly more critical)
    parser.add_argument("--w_mid", type=float, default=0.60)
    parser.add_argument("--w_ext", type=float, default=0.40)

    # Safety constraints
    parser.add_argument("--ext_tolerance", type=float, default=0.10)  # discuss later
    parser.add_argument("--enforce_low", action="store_true")         # default False
    parser.add_argument("--low_tolerance", type=float, default=0.30)  # used only if --enforce_low

    # Early stopping
    parser.add_argument("--patience_evals", type=int, default=8)

    parser.add_argument("--save_dir", type=str, default="pia_runs/run_paramloss_3regime")
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()
    main(args)
