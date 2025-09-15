# compare.py
import numpy as np

for ax in ("x","y"):
    print(f"-------------------------------{ax}-Kick Comparison---------------------------")
    k_pyht = np.load(f'{ax}_kicks_pyht.npy')
    k_xs   = np.load(f'{ax}_kicks_xsuite.npy')

    if k_pyht.shape != k_xs.shape:
        sz = min((len(k_pyht),len(k_xs)))
        k_pyht = k_pyht[:sz]
        k_xs = k_xs[:sz]

    abs_diff = np.abs(k_pyht - k_xs)
    rel_diff = abs_diff / np.maximum(1e-20, np.abs(k_pyht))

    print("Per-passage stats (PyHT vs Xsuite):")
    for i, (a, b, ad, rd) in enumerate(zip(k_pyht, k_xs, abs_diff, rel_diff), 1):
        print(f"  pass {i:02d}: kick_pyht={a:+.6e}, kick_xsuite={b:+.6e}, "
            f"|Δ|={ad:.3e}, rel={rd:.3e}")

    print("\nSummary:")
    print(f"  max |Δ|   = {abs_diff.max():.3e}")
    print(f"  max rel Δ = {rel_diff.max():.3e}")
    print(f"  L2 rel Δ  = {np.linalg.norm(k_pyht - k_xs)/np.linalg.norm(k_pyht):.3e}")
