"""Ajunta diverses xarxes del valor de l'energia (mateixes variables i normalització) en un sol fitxer
amb tots els membres; l'agent fa la mitjana de tots. Ús: ajunta.py <sortida.npz> <xarxa1.npz> <xarxa2.npz> ..."""
import sys
import numpy as np
fs = [np.load(p) for p in sys.argv[2:]]
for f in fs[1:]:
    assert np.allclose(f["mu"], fs[0]["mu"]) and np.allclose(f["sd"], fs[0]["sd"]), "normalització diferent"
D = {k: fs[0][k] for k in ("tipus", "mu", "sd", "n_capes", "ysc")}
i = 0
for f in fs:
    for m in range(int(f["n_membres"])):
        for k in ("W%d_0", "b%d_0", "W%d_1", "b%d_1"):
            D[k % i] = f[k % m]
        i += 1
D["n_membres"] = np.int64(i)
np.savez_compressed(sys.argv[1], **D)
print("membres:", i, "->", sys.argv[1])
