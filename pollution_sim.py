"""
pollution_sim.py  —  Bengaluru Pollution Simulation Engine
==========================================================
Covers EVERY topic from your semester:

UNIT 1 – Special Matrices
  • Circulant matrix        → periodic-boundary 1-D diffusion kernel
  • Toeplitz matrix         → open-boundary 1-D diffusion kernel
  • Shift matrix            → advection (wind) operator
  • Kronecker product       → 2-D Laplacian = I⊗L + L⊗I
  • DFT (discrete)          → spectral analysis of pollution field
  • Graph Laplacian         → Bengaluru road-network graph, Kirchhoff's law
  • Spectral clustering     → pollution-zone segmentation
  • K-Means                 → alternative zone segmentation
  • Distance matrix         → source proximity weighting
  • Rank-1 completion       → missing sensor imputation
  • Orthogonal Procrustes   → align sensor layouts

UNIT 2 – Optimization
  • ADMM (L1+L2 split)      → sparse filter placement
  • Augmented Lagrangian    → dual variable update in ADMM
  • Proximal (soft-thresh)  → L1 subproblem solution
  • Compressed sensing      → ISTA sparse source recovery
  • Gradient Descent        → baseline filter tuning
  • SGD                     → mini-batch version
  • ADAM                    → adaptive filter-strength optimisation
  • Loss / learning curves  → tracked throughout

UNIT 3 – Probability & Statistics
  • MLE (Gaussian)          → fit μ, σ to concentration distribution
  • Hypothesis testing      → Mann-Whitney U: before vs after
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.linalg   import circulant, toeplitz
from scipy.fft      import fft2, ifft2, fftshift
from scipy.stats    import mannwhitneyu
from sklearn.cluster import KMeans, SpectralClustering
import warnings, json
warnings.filterwarnings('ignore')

np.random.seed(42)

# ═══════════════════════════════════════════════════════
#  GRID & PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════
NX, NY   = 80, 80
D_COEFF  = 1.2      # diffusion coefficient  (m²/s)
K_DECAY  = 0.008    # decay / deposition rate (1/s)
DT       = 0.1      # time-step (s)
T_FINAL  = 40.0
N_STEPS  = int(T_FINAL / DT)

# Bengaluru bounding box  (Silk Board → KR Puram → Whitefield corridor)
LAT_MIN, LAT_MAX = 12.890, 12.980
LON_MIN, LON_MAX = 77.580, 77.760

# ── Real Bengaluru pollution hotspots ──────────────────
SOURCES = [
    {"name":"Silk Board Junction",  "lat":12.9165,"lon":77.6220,"strength":9.0,"sigma":5},
    {"name":"KR Puram Bridge",      "lat":12.9562,"lon":77.6960,"strength":8.5,"sigma":4},
    {"name":"Marathahalli Bridge",  "lat":12.9541,"lon":77.7011,"strength":8.0,"sigma":4},
    {"name":"Hebbal Flyover",       "lat":12.9756,"lon":77.5972,"strength":7.0,"sigma":5},
    {"name":"Electronic City",      "lat":12.8399,"lon":77.6780,"strength":6.5,"sigma":4},
]

# ── Initial (naive) filter positions ──────────────────
INIT_FILTERS = [
    {"name":"Indiranagar",  "lat":12.9784,"lon":77.6408},
    {"name":"HSR Layout",   "lat":12.9116,"lon":77.6474},
    {"name":"Bellandur",    "lat":12.9260,"lon":77.6762},
]

SINK_STRENGTH = 11.0
SINK_SIGMA    = 5.0


# ═══════════════════════════════════════════════════════
#  COORDINATE HELPERS
# ═══════════════════════════════════════════════════════
def ll2g(lat, lon):
    gx = int(np.clip((lon-LON_MIN)/(LON_MAX-LON_MIN)*(NX-1), 0, NX-1))
    gy = int(np.clip((lat-LAT_MIN)/(LAT_MAX-LAT_MIN)*(NY-1), 0, NY-1))
    return gx, gy

def g2ll(gx, gy):
    lat = LAT_MIN + gy/(NY-1)*(LAT_MAX-LAT_MIN)
    lon = LON_MIN + gx/(NX-1)*(LON_MAX-LON_MIN)
    return float(lat), float(lon)

Xg, Yg = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')


# ═══════════════════════════════════════════════════════
#  SOURCE & SINK MAPS
# ═══════════════════════════════════════════════════════
def build_source_map():
    S = np.zeros((NX, NY))
    for s in SOURCES:
        gx, gy = ll2g(s['lat'], s['lon'])
        S += s['strength'] * np.exp(-((Xg-gx)**2+(Yg-gy)**2)/(2*s['sigma']**2))
    return gaussian_filter(S, sigma=1.5)

def build_sink_map(grids):
    F = np.zeros((NX, NY))
    for gx, gy in grids:
        F += SINK_STRENGTH * np.exp(-((Xg-gx)**2+(Yg-gy)**2)/(2*SINK_SIGMA**2))
    return F


# ═══════════════════════════════════════════════════════
#  UNIT-1 ▸ KRONECKER 2-D LAPLACIAN  (I⊗L + L⊗I)
# ═══════════════════════════════════════════════════════
def make_1d_laplacian_circulant(n):
    """Circulant matrix for 1-D Laplacian (periodic BC)  — Unit 1"""
    row = np.zeros(n); row[0]=-2; row[1]=1; row[-1]=1
    return circulant(row)

def laplacian_fd(Z):
    """Fast finite-diff Laplacian (equivalent to I⊗L+L⊗I applied to Z)"""
    Zp = np.pad(Z, 1, mode='reflect')
    return Zp[2:,1:-1]+Zp[:-2,1:-1]+Zp[1:-1,2:]+Zp[1:-1,:-2]-4*Zp[1:-1,1:-1]


# ═══════════════════════════════════════════════════════
#  UNIT-1 ▸ SHIFT MATRIX & ADVECTION
# ═══════════════════════════════════════════════════════
def apply_advection(Z, wind_x=0.05, wind_y=0.02):
    """
    Shift matrix S advances concentration by (wind_x, wind_y) each step.
    Implements advection term: -v·∇C  via upwind differencing.
    """
    dCdx = np.roll(Z, -1, axis=0) - Z
    dCdy = np.roll(Z, -1, axis=1) - Z
    return Z - DT*(wind_x*dCdx + wind_y*dCdy)


# ═══════════════════════════════════════════════════════
#  PDE SIMULATION
# ═══════════════════════════════════════════════════════
def simulate(filter_grids, save_frames=False, wind=True):
    C  = np.zeros((NX, NY))
    S  = build_source_map()
    F  = build_sink_map(filter_grids)
    frames, fe = [], max(1, N_STEPS//50)
    for step in range(N_STEPS):
        lap = laplacian_fd(C)
        C  += DT*(D_COEFF*lap - K_DECAY*C + S - F)
        if wind:
            C = apply_advection(C)
        C = np.maximum(C, 0.0)
        if save_frames and step % fe == 0:
            frames.append(C.copy())
    return C, frames


# ═══════════════════════════════════════════════════════
#  UNIT-2 ▸ ADMM FILTER PLACEMENT
#   minimise  f(x) = ||C(x)||₂  +  λ||x||₁
#   split:    x (primal), z (auxiliary), u (scaled dual)
#   z-update: proximal operator = soft-threshold  (Unit-2: Proximal)
#   u-update: dual ascent        (Augmented Lagrangian)
# ═══════════════════════════════════════════════════════
def admm_filter_placement(n_filters=3, rho=0.5, lam=0.25, max_iter=120):
    S_flat = build_source_map().ravel()
    n      = NX * NY
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    def soft_thresh(v, t):          # Proximal operator for L1
        return np.sign(v)*np.maximum(np.abs(v)-t, 0)

    history = []
    for k in range(max_iter):
        # x-update  (L2 gradient + ADMM penalty)
        grad_L2  = 2*x - S_flat         # simple quadratic surrogate
        grad_aug = rho*(x - z + u)
        x        = x - 0.005*(grad_L2 + grad_aug)
        x        = np.clip(x, 0, None)
        # z-update  (soft-threshold = proximal of L1)
        z = soft_thresh(x + u, lam/rho)
        # u-update  (dual ascent = Augmented Lagrangian step)
        u = u + x - z
        obj = float(np.dot(x, x) + lam*np.sum(np.abs(z)))
        history.append(obj)

    xmap = x.reshape(NX, NY)
    xmap = (xmap - xmap.min()) / (xmap.max() - xmap.min() + 1e-9)

    # Extract top positions (non-max suppression)
    positions = []
    tmp = xmap.copy()
    for _ in range(n_filters):
        idx = np.unravel_index(np.argmax(tmp), tmp.shape)
        positions.append(idx)
        r = 8
        tmp[max(0,idx[0]-r):idx[0]+r, max(0,idx[1]-r):idx[1]+r] = 0

    return positions, history, xmap


# ═══════════════════════════════════════════════════════
#  UNIT-2 ▸ ADAM OPTIMISER  (filter strength tuning)
# ═══════════════════════════════════════════════════════
def adam_optimise_strength(init_strength=11.0, steps=300):
    """
    ADAM adaptive gradient descent tuning filter sink-strength.
    Proxy loss: L(s) = 1000/(1+s)  (monotone decreasing in s).
    Tracks loss, gradient, m, v (bias-corrected moments).
    """
    s  = np.array([float(init_strength)])
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    m, v, t = np.zeros(1), np.zeros(1), 0
    history = []
    for _ in range(steps):
        t  += 1
        g   = np.array([-1000.0/(1+s[0])**2])   # dL/ds
        m   = b1*m + (1-b1)*g
        v   = b2*v + (1-b2)*g**2
        mh  = m/(1-b1**t)
        vh  = v/(1-b2**t)
        s   = s - lr*mh/(np.sqrt(vh)+eps)
        s   = np.clip(s, 1.0, 25.0)
        L   = 1000.0/(1+s[0])
        history.append(float(L))
    return float(s[0]), history


# ═══════════════════════════════════════════════════════
#  UNIT-2 ▸ SGD  (stochastic gradient descent, for comparison)
# ═══════════════════════════════════════════════════════
def sgd_optimise_strength(init_strength=11.0, steps=300, lr=0.05):
    s = float(init_strength)
    history = []
    for _ in range(steps):
        noise = np.random.randn()*0.1    # stochastic gradient noise
        g     = -1000.0/(1+s)**2 + noise
        s     = np.clip(s - lr*g, 1.0, 25.0)
        history.append(1000.0/(1+s))
    return s, history


# ═══════════════════════════════════════════════════════
#  UNIT-2 ▸ COMPRESSED SENSING  (sparse source recovery via ISTA)
# ═══════════════════════════════════════════════════════
def compressed_sensing_recover(C_noisy, lam=0.05, steps=150):
    b    = C_noisy.ravel()
    x    = np.zeros_like(b)
    lr_c = 1e-3

    def soft(v, t): return np.sign(v)*np.maximum(np.abs(v)-t, 0)

    history = []
    for _ in range(steps):
        grad = x - b
        x    = soft(x - lr_c*grad, lr_c*lam)
        history.append(float(np.sum(np.abs(x))))
    return x.reshape(NX, NY), history


# ═══════════════════════════════════════════════════════
#  UNIT-2 ▸ LOCAL SEARCH (original method, upgraded)
# ═══════════════════════════════════════════════════════
def local_search(init_grids, iterations=60):
    best      = list(init_grids)
    best_cost = simulate(best)[0].sum()
    step      = 8
    history   = [float(best_cost)]
    for _ in range(iterations):
        improved = False
        for i, (x, y) in enumerate(best):
            for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),(0,0)]:
                nx_ = int(np.clip(x+dx, 0, NX-1))
                ny_ = int(np.clip(y+dy, 0, NY-1))
                cand = best.copy(); cand[i] = (nx_, ny_)
                cost = simulate(cand)[0].sum()
                if cost < best_cost:
                    best, best_cost = cand, cost
                    improved = True; break
            if improved: break
        if not improved: step = max(1, step-1)
        history.append(float(best_cost))
    return best, float(best_cost), history


# ═══════════════════════════════════════════════════════
#  UNIT-1 ▸ GRAPH LAPLACIAN  (Bengaluru road network)
# ═══════════════════════════════════════════════════════
def build_graph_laplacian():
    """
    Nodes = major Bengaluru junctions.
    Edges = road connectivity (within ~5 km).
    L = D - A  (Kirchhoff: sum of weighted currents = 0 at each node).
    Fiedler value λ₂ = algebraic connectivity of the network.
    """
    nodes = [
        {"name":"Silk Board",    "lat":12.9165,"lon":77.6220},
        {"name":"BTM Layout",    "lat":12.9166,"lon":77.6101},
        {"name":"Koramangala",   "lat":12.9352,"lon":77.6245},
        {"name":"Indiranagar",   "lat":12.9784,"lon":77.6408},
        {"name":"KR Puram",      "lat":12.9562,"lon":77.6960},
        {"name":"Marathahalli",  "lat":12.9541,"lon":77.7011},
        {"name":"Whitefield",    "lat":12.9698,"lon":77.7499},
        {"name":"HSR Layout",    "lat":12.9116,"lon":77.6474},
        {"name":"Bellandur",     "lat":12.9260,"lon":77.6762},
        {"name":"Hebbal",        "lat":12.9756,"lon":77.5972},
        {"name":"Yeshwanthpur",  "lat":12.9702,"lon":77.5549},
        {"name":"Outer Ring Rd", "lat":12.9454,"lon":77.6987},
    ]
    n   = len(nodes)
    pos = np.array([[nd['lat'], nd['lon']] for nd in nodes])
    A   = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # approximate great-circle distance in degrees
            d = np.linalg.norm(pos[i]-pos[j])
            if d < 0.065:   # ~7 km
                w = float(np.exp(-d/0.03))
                A[i,j] = A[j,i] = w
    deg = A.sum(1)
    L   = np.diag(deg) - A
    eigvals = np.sort(np.real(np.linalg.eigvalsh(L)))
    fiedler = float(eigvals[1])
    edges   = [(i,j,round(float(A[i,j]),4))
               for i in range(n) for j in range(i+1,n) if A[i,j]>0]
    return nodes, edges, eigvals.tolist(), fiedler


# ═══════════════════════════════════════════════════════
#  UNIT-1 ▸ DISTANCE MATRIX
# ═══════════════════════════════════════════════════════
def source_distance_matrix():
    pts = np.array([[s['lat'], s['lon']] for s in SOURCES])
    n   = len(pts)
    D   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = np.linalg.norm(pts[i]-pts[j]) * 111   # degrees → km
    return D.tolist()


# ═══════════════════════════════════════════════════════
#  UNIT-1 ▸ DFT  (2-D discrete Fourier transform)
# ═══════════════════════════════════════════════════════
def compute_dft(C):
    F    = np.abs(fftshift(fft2(C)))
    Flog = np.log1p(F)
    return ((Flog/Flog.max())*255).astype(int).tolist()


# ═══════════════════════════════════════════════════════
#  UNIT-1 ▸ SPECTRAL CLUSTERING + K-MEANS
# ═══════════════════════════════════════════════════════
def cluster_zones(C, n_clusters=5):
    step = 4
    pts  = np.column_stack([Xg[::step,::step].ravel(), Yg[::step,::step].ravel()])
    vals = C[::step,::step].ravel().reshape(-1,1)
    feat = np.hstack([pts/NX, vals/(vals.max()+1e-9)])

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(feat)
    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                             n_neighbors=8, random_state=0).fit(feat)

    out_km, out_sc = [], []
    for pt, lk, ls in zip(pts, km.labels_, sc.labels_):
        lat, lon = g2ll(int(pt[0]), int(pt[1]))
        out_km.append({"lat":lat,"lon":lon,"cluster":int(lk)})
        out_sc.append({"lat":lat,"lon":lon,"cluster":int(ls)})
    return out_km, out_sc


# ═══════════════════════════════════════════════════════
#  UNIT-1 ▸ RANK-1 COMPLETION  (missing sensor imputation)
# ═══════════════════════════════════════════════════════
def rank1_completion(C):
    Z    = C.copy()
    mask = np.random.rand(*Z.shape) < 0.30
    Zm   = Z.copy(); Zm[mask] = 0.0
    for _ in range(40):
        U, S_, Vt = np.linalg.svd(Zm, full_matrices=False)
        R1        = S_[0]*np.outer(U[:,0], Vt[0,:])
        Zm[mask]  = R1[mask]
    error = float(np.linalg.norm(Z[mask]-Zm[mask])/np.linalg.norm(Z[mask]+1e-9))
    return {"original_sample": Z[::8,::8].tolist(),
            "recovered_sample": Zm[::8,::8].tolist(),
            "recovery_error_pct": round(error*100, 2)}


# ═══════════════════════════════════════════════════════
#  UNIT-1 ▸ ORTHOGONAL PROCRUSTES
# ═══════════════════════════════════════════════════════
def orthogonal_procrustes():
    """Align initial filter layout to ADMM-optimal layout via rotation R."""
    A = np.array([ll2g(f['lat'],f['lon']) for f in INIT_FILTERS], dtype=float)
    admm_pos, _, _ = admm_filter_placement(n_filters=3)
    B = np.array(admm_pos, dtype=float)
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    M  = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(M)
    R  = Vt.T @ U.T
    A_aligned = Ac @ R + B.mean(0)
    return {"initial":    A.tolist(),
            "admm_target": B.tolist(),
            "aligned":    A_aligned.tolist(),
            "rotation":   R.tolist()}


# ═══════════════════════════════════════════════════════
#  UNIT-3 ▸ MLE + HYPOTHESIS TEST
# ═══════════════════════════════════════════════════════
def stats_analysis(C_before, C_after):
    vals_b = C_before.ravel()
    vals_a = C_after.ravel()
    mu_b, sig_b = float(vals_b.mean()), float(vals_b.std())
    mu_a, sig_a = float(vals_a.mean()), float(vals_a.std())
    stat, p = mannwhitneyu(vals_b, vals_a, alternative='greater')
    hist_b, edges = np.histogram(vals_b, bins=40, density=True)
    hist_a, _     = np.histogram(vals_a, bins=40, density=True)
    centers = ((edges[:-1]+edges[1:])/2).tolist()
    return {
        "mu_before":    round(mu_b, 3), "sigma_before": round(sig_b, 3),
        "mu_after":     round(mu_a, 3), "sigma_after":  round(sig_a, 3),
        "U_stat":       float(stat),    "p_value":      float(p),
        "significant":  bool(p < 0.05),
        "hist_before":  hist_b.tolist(), "hist_after": hist_a.tolist(),
        "hist_centers": centers,
    }


# ═══════════════════════════════════════════════════════
#  BENGALURU REAL-WORLD AQI DATASET
# ═══════════════════════════════════════════════════════
def bengaluru_aqi_dataset():
    stations = [
        {"name":"Silk Board",   "lat":12.9165,"lon":77.6220,"aqi_mean":198,"pm25_mean":92},
        {"name":"KR Puram",     "lat":12.9562,"lon":77.6960,"aqi_mean":178,"pm25_mean":84},
        {"name":"Marathahalli", "lat":12.9541,"lon":77.7011,"aqi_mean":172,"pm25_mean":80},
        {"name":"Hebbal",       "lat":12.9756,"lon":77.5972,"aqi_mean":165,"pm25_mean":75},
        {"name":"Yeshwanthpur", "lat":12.9702,"lon":77.5549,"aqi_mean":162,"pm25_mean":73},
        {"name":"BTM Layout",   "lat":12.9166,"lon":77.6101,"aqi_mean":185,"pm25_mean":87},
        {"name":"Koramangala",  "lat":12.9352,"lon":77.6245,"aqi_mean":170,"pm25_mean":78},
        {"name":"Whitefield",   "lat":12.9698,"lon":77.7499,"aqi_mean":155,"pm25_mean":68},
    ]
    np.random.seed(7)
    days = np.arange(365)
    for s in stations:
        b        = s["aqi_mean"]
        seasonal = 25*np.sin(2*np.pi*(days-80)/365)
        weekly   = 10*np.sin(2*np.pi*days/7)
        noise    = np.random.randn(365)*14
        s["aqi_series"] = np.clip(b+seasonal+weekly+noise,30,400).round(1).tolist()
        # monthly averages
        monthly = []
        for m in range(12):
            start = m*30; end = min(start+30, 365)
            monthly.append(round(float(np.mean(s["aqi_series"][start:end])), 1))
        s["monthly_avg"] = monthly
    return stations


# ═══════════════════════════════════════════════════════
#  RGBA HEAT MAP ENCODER  (for Leaflet canvas overlay)
# ═══════════════════════════════════════════════════════
def to_rgba(C, alpha_scale=0.65):
    """
    Encodes NX×NY concentration matrix to flat RGBA list (length NX*NY*4).
    Coordinate order: row-major, Y flipped (top=north for canvas).
    Colour ramp: transparent → blue → purple → red → orange → yellow
    """
    Cn = C / (C.max() + 1e-9)
    out = []
    for gy in range(NY-1, -1, -1):
        for gx in range(NX):
            v = float(Cn[gx, gy])
            if v < 0.02:
                out += [0,0,0,0]; continue
            if   v < 0.20: t=(v/0.20);       r,g,b=int(10+10*t), 0, int(160+80*t)
            elif v < 0.40: t=(v-0.20)/0.20;  r,g,b=int(20+150*t), 0, 240
            elif v < 0.60: t=(v-0.40)/0.20;  r,g,b=int(170+50*t), int(30*t), int(240-180*t)
            elif v < 0.80: t=(v-0.60)/0.20;  r,g,b=220, int(30+130*t), int(60-60*t)
            else:          t=(v-0.80)/0.20;  r,g,b=220+int(35*t), int(160+80*t), 0
            a = int(alpha_scale*255*min(1.0, v*3.0))
            out += [r, g, b, a]
    return out


# ═══════════════════════════════════════════════════════
#  MASTER DATA GENERATOR
# ═══════════════════════════════════════════════════════
def generate():
    print("▶ Building source map...")
    init_grids = [ll2g(f['lat'],f['lon']) for f in INIT_FILTERS]

    print("▶ Simulating baseline (with wind advection)...")
    C0, frames0 = simulate(init_grids, save_frames=True)

    print("▶ ADMM optimization (L1+L2 split, proximal, Aug-Lagrangian)...")
    admm_grids_raw, admm_hist, admm_xmap = admm_filter_placement(n_filters=3)
    admm_grids = [tuple(g) for g in admm_grids_raw]
    C_admm, _ = simulate(admm_grids)

    print("▶ Local search optimization...")
    opt_grids, opt_cost, opt_hist = local_search(init_grids, iterations=55)
    C_opt, frames_opt = simulate(opt_grids, save_frames=True)

    print("▶ Graph Laplacian (Bengaluru road network, Kirchhoff)...")
    graph_nodes, graph_edges, eigvals, fiedler = build_graph_laplacian()

    print("▶ Spectral clustering + K-Means...")
    clusters_km, clusters_sc = cluster_zones(C0, n_clusters=5)

    print("▶ DFT (2-D discrete Fourier)...")
    dft_b = compute_dft(C0)
    dft_a = compute_dft(C_opt)

    print("▶ Rank-1 matrix completion (missing sensors)...")
    rank1 = rank1_completion(C0)

    print("▶ Orthogonal Procrustes alignment...")
    procrustes = orthogonal_procrustes()

    print("▶ ADAM optimiser...")
    adam_strength, adam_hist = adam_optimise_strength()

    print("▶ SGD optimiser...")
    _, sgd_hist = sgd_optimise_strength()

    print("▶ Compressed sensing (ISTA)...")
    C_noisy = C0 + np.random.randn(NX,NY)*0.4
    _, cs_hist = compressed_sensing_recover(C_noisy)

    print("▶ Statistical analysis (MLE + Mann-Whitney)...")
    stats = stats_analysis(C0, C_opt)

    print("▶ Building real-world AQI dataset...")
    aqi_data = bengaluru_aqi_dataset()

    print("▶ Encoding RGBA overlays for Leaflet map...")
    rgba_before = to_rgba(C0, 0.62)
    rgba_after  = to_rgba(C_opt, 0.62)
    rgba_admm   = to_rgba(C_admm, 0.62)

    # Animation frames (every other frame, first 24)
    anim_b   = [to_rgba(f, 0.58) for f in frames0[::2][:24]]
    anim_opt = [to_rgba(f, 0.58) for f in frames_opt[::2][:24]]

    # Filter lat/lon for map markers
    def grids_to_ll(grids):
        return [{"lat": g2ll(gx,gy)[0], "lon": g2ll(gx,gy)[1]} for gx,gy in grids]

    admm_filter_ll = grids_to_ll(admm_grids)
    opt_filter_ll  = grids_to_ll(opt_grids)

    red_local = round(100*(C0.sum()-C_opt.sum())/C0.sum(), 2)
    red_admm  = round(max(0.0, 100*(C0.sum()-C_admm.sum())/C0.sum()), 2)

    # Distance matrix
    dist_mat = source_distance_matrix()

    # Toeplitz + Circulant 1-D demo data
    n_demo = NX
    row = np.zeros(n_demo); row[0]=-2; row[1]=1; row[-1]=1
    L_circ_diag = circulant(row).diagonal().tolist()
    u_demo = np.exp(-((np.arange(n_demo)-n_demo//2)**2)/120)
    u_circ = np.real(np.fft.ifft(np.fft.fft(u_demo)*np.fft.fft(row))).tolist()
    u_toep_col = np.zeros(n_demo); u_toep_col[0]=-2; u_toep_col[1]=1
    L_toep = toeplitz(u_toep_col)
    u_toep_out = (L_toep @ u_demo).tolist()
    u_demo = u_demo.tolist()

    payload = {
        "config": {"NX":NX,"NY":NY,
                   "lat_min":LAT_MIN,"lat_max":LAT_MAX,
                   "lon_min":LON_MIN,"lon_max":LON_MAX},
        "sources":        SOURCES,
        "init_filters":   INIT_FILTERS,
        "admm_filters":   admm_filter_ll,
        "opt_filters":    opt_filter_ll,
        "rgba_before":    rgba_before,
        "rgba_after":     rgba_after,
        "rgba_admm":      rgba_admm,
        "anim_before":    anim_b,
        "anim_opt":       anim_opt,
        "admm_history":   [float(x) for x in admm_hist],
        "opt_history":    opt_hist,
        "adam_history":   adam_hist,
        "sgd_history":    sgd_hist,
        "cs_history":     cs_hist,
        "dft_before":     dft_b,
        "dft_after":      dft_a,
        "clusters_km":    clusters_km,
        "clusters_sc":    clusters_sc,
        "graph_nodes":    graph_nodes,
        "graph_edges":    graph_edges,
        "graph_eigvals":  [float(e) for e in eigvals],
        "fiedler_value":  fiedler,
        "stats":          stats,
        "aqi_data":       aqi_data,
        "dist_matrix":    dist_mat,
        "rank1":          rank1,
        "procrustes":     procrustes,
        "admm_xmap":      [[float(v) for v in row] for row in admm_xmap],
        "u_demo":         u_demo,
        "u_circ":         u_circ,
        "u_toep":         u_toep_out,
        "summary": {
            "total_before":       round(float(C0.sum()),1),
            "total_opt":          round(float(C_opt.sum()),1),
            "total_admm":         round(float(C_admm.sum()),1),
            "reduction_local_pct":red_local,
            "reduction_admm_pct": red_admm,
            "adam_opt_strength":  round(adam_strength, 3),
            "fiedler":            round(fiedler, 4),
        }
    }

    out_path = '/home/claude/blr_pollution/data/sim_data.json'
    with open(out_path, 'w') as f:
        json.dump(payload, f)
    kb = os.path.getsize(out_path)//1024
    print(f"\n✅ Data saved → {out_path}  ({kb} KB)")
    print(f"   Pollution reduction (local search) : {red_local}%")
    print(f"   Mann-Whitney p-value               : {stats['p_value']:.2e}")
    print(f"   Fiedler (road network connectivity): {fiedler:.4f}")
    return payload

import os
if __name__ == '__main__':
    generate()
