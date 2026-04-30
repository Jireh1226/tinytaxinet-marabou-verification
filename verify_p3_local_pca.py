"""
P3 verification under per-image local-PCA regions.

Construction:
  For each sampled centered image x0:
    1. Find same-state neighbors:  |dCTE| <= 0.5, |dHE| <= 5.
    2. Compute D = neighbors - x0  (deviation matrix).
    3. SVD: D = U S V^T. Keep top d=8 directions (rows of V_top).
    4. Per-direction projection std sigma_j = std of D @ V_top[j].
    5. Region: x = x0 + V_top^T @ z  with  |z_j| <= ALPHA * sigma_j,
       intersected with  0 <= x_i <= 1  and  sum(x_i) = 64.

We encode this in Marabou by treating x as the input and adding
128 - d orthogonal-complement equalities  V_null @ x = V_null @ x0
(forces x to lie in the d-dim affine subspace through x0), plus the
per-pixel envelope box implied by the |z_j| <= ALPHA*sigma_j bounds,
plus the mean=0.5 equality.

Verification: for each of 20 images, run the four P3 safety sub-queries
(|CTE| < 10, |HE| < 90).  An image is verified if all four are UNSAT.

We sweep ALPHA in {1, 2, 3} so the report can quote a reachable range
of the local-PCA region (1-sigma, 2-sigma, 3-sigma along each of the 8
PCA directions) and show how the verification rate scales.
"""
import json
import time

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_p3_local_pca.json"

NUM_PIXELS = 128
SUM_TARGET = 0.5 * NUM_PIXELS
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0
P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
P3_CENTERED_THRESHOLD = 1.0
NUM_IMAGES = 20

D = 8                    # number of PCA directions
CTE_BIN_HALF = 0.5       # m
HE_BIN_HALF = 5.0        # deg
ALPHA_VALUES = [1.0, 2.0, 3.0]
TIMEOUT_S = 300


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64), \
               f["X_train"][:].astype(np.float64)


def select_centered_sample(labels):
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]
    return centered_idx[np.linspace(0, len(centered_idx) - 1, NUM_IMAGES, dtype=int)]


def build_local_pca(images, labels, idx, d=D):
    """Return (x0, V_top, sigma_proj, V_null, n_neighbors, expl_var)."""
    cte = labels[:, 0]
    he = labels[:, 1]
    x0 = images[int(idx)]
    cte_q, he_q = cte[int(idx)], he[int(idx)]
    mask = (np.abs(cte - cte_q) <= CTE_BIN_HALF) & \
           (np.abs(he - he_q) <= HE_BIN_HALF)
    mask[int(idx)] = False
    neighbors = images[mask]
    m = len(neighbors)
    if m < d + 1:
        return None
    D_mat = neighbors - x0
    # SVD: D_mat (m, 128) = U (m, k) diag(S) V^T (k, 128), k = min(m, 128)
    _, S, Vt = np.linalg.svd(D_mat, full_matrices=False)
    V_top = Vt[:d]                 # (d, 128)  unit rows in pixel space
    V_null = Vt[d:]                # (k - d, 128) orthogonal complement (within row-space of D_mat)

    proj = D_mat @ V_top.T         # (m, d)
    sigma_proj = proj.std(axis=0)  # (d,)

    expl_var = float((S[:d] ** 2).sum() / (S ** 2).sum())
    return {
        "x0": x0,
        "V_top": V_top,
        "sigma_proj": sigma_proj,
        "V_null": V_null,
        "n_neighbors": int(m),
        "explained_variance_ratio_top_d": expl_var,
        "singular_values": S.tolist(),
    }


def feasibility(x0, V_top, alphas):
    """Per-pixel envelope of x = x0 + V_top^T z, |z_j| <= alpha_j."""
    half_widths = np.sum(np.abs(V_top.T) * alphas[None, :], axis=1)
    pixel_lo = np.maximum(PIXEL_MIN, x0 - half_widths)
    pixel_hi = np.minimum(PIXEL_MAX, x0 + half_widths)
    return pixel_lo, pixel_hi


def solve_one(x0, V_top, V_null, alphas, out_var, direc, bound):
    pixel_lo, pixel_hi = feasibility(x0, V_top, alphas)
    if np.any(pixel_lo > pixel_hi):
        return {"status": "region_empty_after_clip"}

    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]

    # Per-pixel envelope (axis-aligned bounding box of the d-dim region after clipping)
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_lo[i]))
        net.setUpperBound(input_vars[i], float(pixel_hi[i]))

    # Auxiliary latent variables z_1, ..., z_d  with bounds  |z_j| <= alpha_j.
    # Note: V_top has shape (d, 128); rows are unit-norm pixel-space directions.
    # We define   x_i = x0_i + sum_j V_top[j, i] * z_j   via 128 sparse equalities.
    d = V_top.shape[0]
    z_vars = [int(net.getNewVariable()) for _ in range(d)]
    for j in range(d):
        net.setLowerBound(z_vars[j], float(-alphas[j]))
        net.setUpperBound(z_vars[j], float(alphas[j]))

    # Sparse equality per pixel:  x_i  -  sum_j V_top[j, i] z_j  =  x0_i
    for i in range(NUM_PIXELS):
        col = V_top[:, i]
        nz = np.where(np.abs(col) > 1e-12)[0]
        vars_eq = [input_vars[i]] + [z_vars[j] for j in nz]
        coeffs_eq = [1.0] + [float(-col[j]) for j in nz]
        net.addEquality(vars_eq, coeffs_eq, float(x0[i]))

    # Mean constraint  sum(x) = 64
    net.addEquality(list(input_vars), [1.0] * NUM_PIXELS, SUM_TARGET)

    # Negated postcondition
    if direc == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    status = status.strip().lower()

    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        cx = np.array([vals[v] for v in input_vars])
        out["counterexample"] = {
            "CTE": round(float(vals[192]), 6),
            "HE": round(float(vals[193]), 6),
            "input_mean": round(float(cx.mean()), 6),
            "input_min": round(float(cx.min()), 6),
            "input_max": round(float(cx.max()), 6),
            "linf_from_x0": round(float(np.max(np.abs(cx - x0))), 6),
        }
    return out


def main():
    images, labels = load_data()
    sample = select_centered_sample(labels)
    cte = labels[:, 0]

    sub_queries = [
        ("a", 192, ">=",  P3_CTE_BOUND),
        ("b", 192, "<=", -P3_CTE_BOUND),
        ("c", 193, ">=",  P3_HE_BOUND),
        ("d", 193, "<=", -P3_HE_BOUND),
    ]

    out = {
        "d": D,
        "alpha_values": ALPHA_VALUES,
        "neighbor_bins": {"cte_half_width": CTE_BIN_HALF,
                          "he_half_width": HE_BIN_HALF},
        "results_by_alpha": {},
        "per_image_pca_meta": [],
    }

    # Precompute PCA per image once
    pcas = []
    print(f"Building local PCA (d={D}) for {NUM_IMAGES} centered images...\n", flush=True)
    for k, idx in enumerate(sample):
        info = build_local_pca(images, labels, idx)
        if info is None:
            print(f"  img{k+1:>2} idx={idx} -- not enough neighbors")
            pcas.append(None)
            continue
        meta = {
            "image_idx": int(idx),
            "n_neighbors": info["n_neighbors"],
            "explained_variance_ratio_top_d": round(info["explained_variance_ratio_top_d"], 4),
            "sigma_proj": [round(float(s), 5) for s in info["sigma_proj"]],
            "top_singular_values": [round(float(s), 4) for s in info["singular_values"][:D]],
        }
        out["per_image_pca_meta"].append(meta)
        pcas.append(info)
        print(f"  img{k+1:>2} idx={idx}  m={info['n_neighbors']:>3}  "
              f"expl_var(top {D}) = {info['explained_variance_ratio_top_d']:.3f}  "
              f"sigma_proj = [{info['sigma_proj'][0]:.4f}..{info['sigma_proj'][-1]:.4f}]", flush=True)

    print()
    for alpha in ALPHA_VALUES:
        print(f"=== ALPHA = {alpha} sigma ===", flush=True)
        per_image = {}
        verified = violated = empty = 0
        t_alpha = time.time()
        for k, idx in enumerate(sample):
            info = pcas[k]
            if info is None:
                empty += 1
                per_image[f"image_{k+1}"] = {"dataset_index": int(idx),
                                             "skipped": "no PCA"}
                continue
            alphas = alpha * info["sigma_proj"]
            sub_results = {}
            all_unsat = True
            empty_region = False
            for tag, var, direc, b in sub_queries:
                r = solve_one(info["x0"], info["V_top"], info["V_null"],
                              alphas, var, direc, b)
                sub_results[tag] = r
                if r["status"] == "region_empty_after_clip":
                    empty_region = True
                if r["status"] != "unsat":
                    all_unsat = False
            entry = {
                "dataset_index": int(idx),
                "n_neighbors": info["n_neighbors"],
                "alphas_first_last": [round(float(alphas[0]), 5),
                                       round(float(alphas[-1]), 5)],
                "sub_queries": sub_results,
                "verified": bool(all_unsat),
                "region_empty": empty_region,
            }
            per_image[f"image_{k+1}"] = entry
            if empty_region:
                empty += 1
                tag = "EMPTY"
            elif all_unsat:
                verified += 1
                tag = "VERIFIED"
            else:
                violated += 1
                tag = "VIOLATED"
            statuses = " ".join(sub_results[t]["status"][:3] for t, *_ in sub_queries)
            print(f"  img{k+1:>2} idx={idx} {tag:<9} [{statuses}]", flush=True)
        out["results_by_alpha"][f"alpha_{alpha}"] = {
            "verified": verified,
            "violated": violated,
            "empty_region": empty,
            "total": NUM_IMAGES,
            "elapsed_seconds": round(time.time() - t_alpha, 2),
            "per_image": per_image,
        }
        print(f"  => {verified}/{NUM_IMAGES} verified, "
              f"{violated}/{NUM_IMAGES} violated, "
              f"{empty}/{NUM_IMAGES} empty  "
              f"({time.time() - t_alpha:.1f}s)\n", flush=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
