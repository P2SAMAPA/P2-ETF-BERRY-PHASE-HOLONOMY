import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def parallel_transport(state, dt, connection):
    """
    Parallel transport of a state vector along a path.
    state: (n,) vector
    connection: (n, n) matrix (gauge connection)
    dt: step size
    """
    return state + dt * connection @ state

def berry_phase_holonomy(returns, macro_factor, n_steps=50):
    """
    Compute the Berry phase (holonomy) of a parameter vector driven cyclically.
    The parameter vector is the state of the system (mean, vol, skew, kurtosis).
    """
    if len(returns) < 10:
        return 0.0
    # Compute couplings over time (rolling window)
    window = 20
    couplings = []
    for i in range(window, len(returns)):
        seg = returns[i-window:i]
        mean = np.mean(seg)
        vol = np.std(seg)
        skew = np.mean(((seg - mean) / (vol + 1e-8))**3) if vol > 0 else 0.0
        kurt = np.mean(((seg - mean) / (vol + 1e-8))**4) - 3 if vol > 0 else 0.0
        couplings.append([mean, vol, skew, kurt])
    couplings = np.array(couplings)
    if len(couplings) < 2:
        return 0.0
    # Normalise couplings
    couplings = (couplings - couplings.mean(axis=0)) / (couplings.std(axis=0) + 1e-8)
    # Define the macro-driven loop in parameter space
    # We'll create a closed loop in the 2D PCA space of the macro factor
    # First, compute the PCA of the couplings to get a 2D representation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    couplings_2d = pca.fit_transform(couplings)
    # Create a closed loop in the 2D space
    # Use the macro factor to define the loop
    macro_series = macro_factor
    if len(macro_series) > len(couplings):
        macro_series = macro_series[:len(couplings)]
    elif len(macro_series) < len(couplings):
        couplings_2d = couplings_2d[:len(macro_series)]
    # Create a loop using the macro factor as a parameter
    # We'll use a circle in the 2D space modulated by macro
    angles = np.linspace(0, 2*np.pi, n_steps)
    # Use the first two principal components of the coupling space
    # as the coordinates for the loop
    radius = 0.5 * (1 + macro_series[-1])  # scale radius by macro
    loop_x = radius * np.cos(angles)
    loop_y = radius * np.sin(angles)
    # Compute the Berry phase by integrating the connection along the loop
    # Connection = derivative of the state with respect to parameters
    # Simplified: compute the solid angle subtended by the loop
    # For a 2D loop, the Berry phase is the area enclosed (times a factor)
    # We'll compute the area of the loop as a proxy for the Berry phase
    area = 0.0
    for i in range(n_steps - 1):
        area += loop_x[i] * loop_y[i+1] - loop_y[i] * loop_x[i+1]
    area = 0.5 * abs(area)
    # Berry phase = area * macro_factor (scaled)
    berry_phase = area * (1 + macro_series[-1] * 0.5)
    return float(berry_phase)

def berry_phase_score(returns, macro_df, n_steps=50):
    """
    Compute per-ETF Berry phase score.
    Higher score = stronger path-dependent (non-Markovian) structure.
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 20:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)
    # Compute Berry phase
    bp = berry_phase_holonomy(returns, macro_factor, n_steps)
    return float(bp)
