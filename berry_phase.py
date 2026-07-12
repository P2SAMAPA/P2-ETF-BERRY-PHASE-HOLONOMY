import numpy as np
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

def berry_phase_score(returns, macro_df, n_steps=50):
    """
    Compute per-ETF Berry phase score using curvature of the path.
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
    # Compute rolling couplings (mean, vol, skew, kurtosis)
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
    if len(couplings) < 5:
        return 0.0
    # Normalise couplings
    couplings = (couplings - couplings.mean(axis=0)) / (couplings.std(axis=0) + 1e-8)
    # Compute curvature: second derivative of the path
    # Use the first principal component of couplings as the path
    pca_c = PCA(n_components=1)
    path = pca_c.fit_transform(couplings).flatten()
    # Compute first and second derivatives
    dt = 1.0
    d1 = np.gradient(path, dt)
    d2 = np.gradient(d1, dt)
    # Berry phase proxy: sum of absolute curvature along the path
    curvature = np.sum(np.abs(d2)) / len(d2)
    # Scale by macro factor
    macro_scalar = 1.0 + macro_factor[-1] * 0.5
    score = curvature * macro_scalar
    return float(score)
