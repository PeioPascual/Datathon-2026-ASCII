"""Shared training utilities (offline only): ridge / reduced-rank ridge / numpy MLP with Adam, exports to the
parameter dicts read by agent.prediu_nt."""
import numpy as np


def standard(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    return mu, sd


def ridge_fit(Z, Y, alpha, w=None):
    """Z standardised (n, f), Y (n, o) -> W, b. Intercept unpenalised."""
    if w is None:
        w = np.ones(len(Z))
    ym = (w[:, None] * Y).sum(0) / w.sum(); zm = (w[:, None] * Z).sum(0) / w.sum()
    Zc, Yc = Z - zm, Y - ym
    A = (Zc * w[:, None]).T @ Zc + alpha * np.eye(Z.shape[1])
    W = np.linalg.solve(A, (Zc * w[:, None]).T @ Yc)
    return W, ym - zm @ W


def rrr(Z, W, b, r):
    """Reduced-rank projection of a ridge fit: keep the top-r directions of the fitted outputs."""
    F = Z @ W
    F = F - F.mean(0)
    _, _, Vt = np.linalg.svd(F, full_matrices=False)
    V = Vt[:r].T
    Wr = W @ V @ V.T
    br = b + (Z @ W).mean(0) @ (np.eye(len(b)) - V @ V.T)
    return Wr, br


def lin_dict(mu, sd, W, b, s2):
    return dict(tipus=np.array("lin"), mu=mu, sd=sd, W=W, b=b, s2=s2)


def mlp_train(Z, Y, Zv, Yv, hidden=(32,), wd=1e-3, lr=3e-3, epochs=300, batch=256, seed=0, patience=25, drop_in=0.0, M=None, Mv=None):
    """Y already standardised. Masked MSE (M = 1 where target known). Early stopping on (Zv, Yv)."""
    rng = np.random.default_rng(seed)
    sizes = [Z.shape[1], *hidden, Y.shape[1]]
    Ws = [rng.normal(0, 1 / np.sqrt(a), (a, b_)) for a, b_ in zip(sizes[:-1], sizes[1:])]
    bs = [np.zeros(b_) for b_ in sizes[1:]]
    if M is None: M = np.ones_like(Y)
    if Mv is None: Mv = np.ones_like(Yv)
    params = Ws + bs
    m1 = [np.zeros_like(p) for p in params]; m2 = [np.zeros_like(p) for p in params]
    t = 0; best = (np.inf, None, 0)
    L = len(Ws)

    def fwd(X, Ws, bs):
        hs = [X]
        for i in range(L):
            h = hs[-1] @ Ws[i] + bs[i]
            hs.append(np.tanh(h) if i < L - 1 else h)
        return hs

    for ep in range(epochs):
        idx = rng.permutation(len(Z))
        for s in range(0, len(Z), batch):
            j = idx[s:s + batch]
            Xb = Z[j]
            if drop_in > 0:
                Xb = Xb * (rng.random(Xb.shape) > drop_in) / (1 - drop_in)
            hs = fwd(Xb, Ws, bs)
            g = (hs[-1] - Y[j]) * M[j] * 2 / max(M[j].sum(), 1) * Y.shape[1] / Y.shape[1]
            gW, gb = [None] * L, [None] * L
            for i in range(L - 1, -1, -1):
                gW[i] = hs[i].T @ g + wd * Ws[i]
                gb[i] = g.sum(0)
                if i > 0:
                    g = (g @ Ws[i].T) * (1 - hs[i] ** 2)
            grads = gW + gb
            t += 1
            for p, gr, a, b2 in zip(params, grads, m1, m2):
                a *= 0.9; a += 0.1 * gr
                b2 *= 0.999; b2 += 0.001 * gr * gr
                p -= lr * (a / (1 - 0.9 ** t)) / (np.sqrt(b2 / (1 - 0.999 ** t)) + 1e-8)
        pv = fwd(Zv, Ws, bs)[-1]
        lv = float((((pv - Yv) ** 2) * Mv).sum() / Mv.sum())
        if lv < best[0] - 1e-5:
            best = (lv, ([w.copy() for w in Ws], [b_.copy() for b_ in bs]), ep)
        elif ep - best[2] > patience:
            break
    return best


def mlp_predict(Ws, bs, Z):
    h = Z
    for i in range(len(Ws)):
        h = h @ Ws[i] + bs[i]
        if i < len(Ws) - 1:
            h = np.tanh(h)
    return h


def mlp_dict(mu, sd, ymu, ysd, members, s2):
    d = dict(tipus=np.array("mlp"), mu=mu, sd=sd, ymu=ymu, ysd=ysd, n_membres=np.int32(len(members)),
             n_capes=np.int32(len(members[0][0])), s2=s2)
    for mi, (Ws, bs) in enumerate(members):
        for i, (W, b) in enumerate(zip(Ws, bs)):
            d["W%d_%d" % (mi, i)] = W; d["b%d_%d" % (mi, i)] = b
    return d
