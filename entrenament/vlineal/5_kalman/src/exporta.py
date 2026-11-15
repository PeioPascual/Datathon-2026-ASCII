"""Exporta l'ajust del filtre de Kalman (data/kf_<plec>_ml.npz) al fitxer que llegeix l'agent (kf_mercat.npz).
Ús: exporta.py <entrada.npz> <sortida kf_mercat.npz>"""
import sys
import numpy as np
z = np.load(sys.argv[1])
np.savez(sys.argv[2], **{k: z[k].astype(float) for k in ("phi", "Q", "H", "mu", "lam")})
print("Kalman exportat ->", sys.argv[2])
