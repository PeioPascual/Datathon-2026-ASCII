"""Ridge del mercat amb mitjana de pesos sobre una graella de regularitzacions (agent R4: 'aavgA').
Treu la tria discreta de l'alpha, que movia la nota +-0.05 entre reentrenaments. Ús: aavg.py <dataset.npz> <sortida.npz>"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ntlib as nl
d = np.load(sys.argv[1]); X, Y = d["X"], d["Y"]
tr = ~d["fora"]; fit = tr & ~np.isnan(Y).any(1)
mu, sd = nl.standard(X[tr]); Z = (X - mu) / sd
ALPHAS = (100, 180, 300, 550, 1000, 1800, 3000)
Ws = [nl.ridge_fit(Z[fit], Y[fit], float(a)) for a in ALPHAS]
W = np.mean([w for w, _ in Ws], 0); b = np.mean([x for _, x in Ws], 0)
s2 = ((Z[fit] @ W + b - Y[fit]) ** 2).mean(0)
np.savez_compressed(sys.argv[2], **nl.lin_dict(mu, sd, W, b, s2))
print("aavg", len(ALPHAS), "alphes, files", int(fit.sum()), "MSE in-sample", round(float(s2.mean()), 5))
