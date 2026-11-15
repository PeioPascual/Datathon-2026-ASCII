"""Validation windows: the 4 standard folds (final d01 d08 d15) and SHIFTED windows named <fold>s<k> (+k days) or
<fold>m<k> (-k days), e.g. d01s3 = every month's days 4..10. A shifted week keeps the same calendar month as its fold
week (days outside the month are dropped) and is kept only with >= MIN_HORES_SETMANA hours (same rule as Q3 sw2.py).
dies_plec(<window>, marge) = every evaluated day +- marge -> what each artefact of that window must exclude."""
import re, sys
sys.path.insert(0, "local_eval")
import pandas as pd
import dades

_RX = re.compile(r"^(final|d01|d08|d15)([sm])(\d+)$")


def setmanes(df, plec):
    r = _RX.match(plec)
    if not r:
        return dades.setmanes(df, plec)
    base, sg, k = r.group(1), r.group(2), int(r.group(3))
    k = k if sg == "s" else -k
    out = []
    for b in dades.setmanes(df, base):
        d0 = b.index[0].normalize() + pd.Timedelta(days=k)
        mes = df[df.index.to_period("M") == b.index[0].to_period("M")]
        dies = mes.index.normalize()
        bb = mes[(dies >= d0) & (dies <= d0 + pd.Timedelta(days=6))]
        if len(bb) >= dades.MIN_HORES_SETMANA:
            out.append(bb)
    return out


def dies_plec(df, plec, marge_dies=0):
    if plec == "tot":
        return []
    dies = set()
    for b in setmanes(df, plec):
        dies.update(pd.date_range(b.index.min().normalize() - pd.Timedelta(days=marge_dies),
                                  b.index.max().normalize() + pd.Timedelta(days=marge_dies)))
    return list(dies)


def conjunt_validacio(df, plec):
    return pd.concat([b.assign(setmana=i) for i, b in enumerate(setmanes(df, plec))])
