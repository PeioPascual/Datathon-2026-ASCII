import numpy as np, math


def filt(y, Z, phi, Q, H, mu, a0=None, P0=None, keep=False):
    """Same as kf.filt but faster: both series observed or both missing (true in these data)."""
    N = len(y)
    PP = np.outer(phi, phi)
    a = np.zeros(8) if a0 is None else a0.copy()
    P = Q / (1 - PP) if P0 is None else P0.copy()
    ll = 0.0; nobs = 0
    A = np.empty((N, 8)) if keep else None
    PS = np.empty((N, 8, 8)) if keep == 2 else None
    obs = ~np.isnan(y).any(1)
    yc = y - mu
    h0, h1 = float(H[0]), float(H[1])
    L2P = math.log(2 * math.pi)
    for t in range(N):
        if t:
            a = phi * a
            P = PP * P + Q
        if obs[t]:
            Zt = Z[t]
            v = yc[t] - Zt @ a
            PZ = P @ Zt.T
            F = Zt @ PZ
            f00 = F[0, 0] + h0; f11 = F[1, 1] + h1; f01 = F[0, 1]
            det = f00 * f11 - f01 * f01
            Fi = np.array([[f11, -f01], [-f01, f00]]) / det
            K = PZ @ Fi
            a = a + K @ v
            P = P - K @ PZ.T
            v0, v1 = v
            ll -= 0.5 * (math.log(det) + (v0 * v0 * f11 - 2 * v0 * v1 * f01 + v1 * v1 * f00) / det + 2 * L2P)
            nobs += 2
        if keep:
            A[t] = a
            if keep == 2:
                PS[t] = P
    return ll, nobs, A, PS


def filt_batch(y, Z, phi, Q, H, mu, keep=False):
    """B parameter sets at once. phi (B,8), Q (B,8,8), H (B,2). Returns ll (B,), nobs, A (B,N,8) if keep."""
    B = phi.shape[0]; N = len(y)
    PP = phi[:, :, None] * phi[:, None, :]
    a = np.zeros((B, 8))
    P = Q / (1 - PP)
    ll = np.zeros(B); nobs = 0
    A = np.empty((B, N, 8)) if keep else None
    obs = ~np.isnan(y).any(1)
    yc = y - mu
    L2P = 2 * math.log(2 * math.pi)
    for t in range(N):
        if t:
            a = phi * a
            P = PP * P + Q
        if obs[t]:
            Zt = Z[t]
            v = yc[t][None, :] - a @ Zt.T                   # (B,2)
            PZ = P @ Zt.T                                   # (B,8,2)
            F = np.einsum("sk,bkr->bsr", Zt, PZ)
            f00 = F[:, 0, 0] + H[:, 0]; f11 = F[:, 1, 1] + H[:, 1]; f01 = F[:, 0, 1]
            det = f00 * f11 - f01 * f01
            Fi = np.stack([np.stack([f11, -f01], 1), np.stack([-f01, f00], 1)], 1) / det[:, None, None]
            K = PZ @ Fi                                     # (B,8,2)
            a = a + np.einsum("bkr,br->bk", K, v)
            P = P - K @ PZ.transpose(0, 2, 1)
            v0, v1 = v[:, 0], v[:, 1]
            ll -= 0.5 * (np.log(det) + (v0 * v0 * f11 - 2 * v0 * v1 * f01 + v1 * v1 * f00) / det + L2P)
            nobs += 2
        if keep:
            A[:, t] = a
    return ll, nobs, A
