"""Structural state-space model of the (CO2, price) log-anomaly pair and its Kalman filter (pure numpy).
State (8): [L_c, L_p, F1_c, F1_p, F2_c, F2_p, S_c, S_p]; each component is an AR(1) with its own persistence per series;
innovations of the same component are correlated across the two series (2x2 block of Q).
Observation: y_s(t) = mu_s + L_s + lam1_s(h) F1_s + lam2_s(h) F2_s + S_s + eps_s."""
import numpy as np

NC = 4  # components: level, shape1, shape2, short


def unpack(th):
    th = np.asarray(th, float)
    phi = 1 / (1 + np.exp(-th[:8]))                       # order = state order
    sd = np.exp(th[8:16]).reshape(NC, 2)
    rho = np.tanh(th[16:20])
    Q = np.zeros((8, 8))
    for c in range(NC):
        i, j = 2 * c, 2 * c + 1
        Q[i, i], Q[j, j] = sd[c, 0] ** 2, sd[c, 1] ** 2
        Q[i, j] = Q[j, i] = rho[c] * sd[c, 0] * sd[c, 1]
    H = np.exp(2 * th[20:22])
    return phi, Q, H


def pack(phi, sd, rho, sde):
    phi = np.clip(np.asarray(phi, float), 1e-4, 1 - 1e-6)
    return np.concatenate([np.log(phi / (1 - phi)), np.log(np.asarray(sd, float).ravel()), np.arctanh(rho), np.log(sde)])


def zmat(lam, hours):
    """lam: (2 series, 2 factors, 24). Returns Z (len(hours), 2, 8)."""
    Z = np.zeros((len(hours), 2, 8))
    for s in (0, 1):
        Z[:, s, s] = 1.0
        Z[:, s, 2 + s] = lam[s, 0, hours]
        Z[:, s, 4 + s] = lam[s, 1, hours]
        Z[:, s, 6 + s] = 1.0
    return Z


def stationary(phi, Q):
    return Q / (1 - np.outer(phi, phi))


def filt(y, Z, phi, Q, H, mu, a0=None, P0=None, keep=False):
    """y (N, 2) with NaN = missing. Returns loglik and (if keep) filtered a (N,8), P (N,8,8) after update at t."""
    N = len(y)
    PP = np.outer(phi, phi)
    a = np.zeros(8) if a0 is None else a0.copy()
    P = stationary(phi, Q) if P0 is None else P0.copy()
    ll = 0.0; nobs = 0
    A = np.zeros((N, 8)) if keep else None
    PS = np.zeros((N, 8, 8)) if keep else None
    first = True
    for t in range(N):
        if not first:
            a = phi * a
            P = PP * P + Q
        first = False
        yt = y[t]
        ok = yt == yt
        if ok.any():
            Zt = Z[t][ok]
            v = yt[ok] - mu[ok] - Zt @ a
            PZ = P @ Zt.T
            F = Zt @ PZ + np.diag(H[ok])
            Fi = np.linalg.inv(F)
            K = PZ @ Fi
            a = a + K @ v
            P = P - K @ PZ.T
            ll += -0.5 * (np.log(np.linalg.det(F)) + v @ Fi @ v + ok.sum() * np.log(2 * np.pi))
            nobs += ok.sum()
        if keep:
            A[t] = a; PS[t] = P
    return ll, nobs, A, PS


def forecast(a, P, Zf, phi, Q, H, mu, var=False):
    """From filtered (a, P) at time t: mean (h, 2) for t+1..t+h given Zf (h, 2, 8); optionally variances (h, 2) and the
    full (h*2) covariance of the forecast vector."""
    h = len(Zf)
    pw = phi[None, :] ** np.arange(1, h + 1)[:, None]          # (h, 8)
    m = mu[None, :] + np.einsum("hsk,hk->hs", Zf, pw * a[None, :])
    if not var:
        return m
    # state covariance at each horizon and cross-horizon covariances: Cov(x_{t+i}, x_{t+j}) = Phi^{j-i} V_i (j>=i)
    PP = np.outer(phi, phi)
    V = []; Vi = P.copy()
    for i in range(h):
        Vi = PP * Vi + Q
        V.append(Vi)
    C = np.zeros((h * 2, h * 2))
    for i in range(h):
        for j in range(i, h):
            Cij = (phi ** (j - i))[None, :] * V[i]           # Cov(x_i, x_j) = V_i Phi^{(j-i)}'  (diag Phi)
            B = Zf[i] @ Cij @ Zf[j].T
            if i == j:
                B = B + np.diag(H)
            C[2 * i:2 * i + 2, 2 * j:2 * j + 2] = B
            C[2 * j:2 * j + 2, 2 * i:2 * i + 2] = B.T
    return m, C
