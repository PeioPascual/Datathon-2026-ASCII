"""rec.pkl -> llarg.pkl: taula llarga d'errors (una fila per prediccio i horitzo k).
Us: llarg.py <rec.pkl> <llarg.pkl>"""
import sys, pickle, numpy as np, pandas as pd

d = pickle.load(open(sys.argv[1], "rb"))
R = d["reals"].set_index("clau")
L = R.load.to_dict(); PV = R.pv.to_dict(); DIA = R.dia.to_dict()
files = []
for r in d["rec"]:
    for k in range(24):
        c = r["clau"] + k
        if c not in L:
            continue
        h = (r["hora"] + k) % 24
        dw = (r["dow"] + (r["hora"] + k) // 24) % 7
        files.append((r["pas"], r["clau"], k, h, r["mes"], (r["cap"][k] if "cap" in r else int(dw >= 5)), r["c"][k], r["s"][k], L[c], PV[c], r["env"][k],
                      r["envmax"], r["k_avui"], r["lp"] if r["lp"] is not None else np.nan, DIA[r["clau"]] if r["clau"] in DIA else DIA[c]))
df = pd.DataFrame(files, columns=["pas", "clau", "k", "h", "mes", "cap", "Lh", "Ph", "L", "P", "env", "envmax", "kav", "lp", "dia"])
df = df[df.L > 1.0]
df.to_pickle(sys.argv[2])
print("files", len(df))
