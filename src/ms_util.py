"""Funções comuns de estimação Markov-switching (MS-AR(1), 2 regimes)."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")


def par(res) -> pd.Series:
    return pd.Series(np.asarray(res.params), index=res.model.param_names)


def pv(res) -> pd.Series:
    return pd.Series(np.asarray(res.pvalues), index=res.model.param_names)


def valido(r, s2_lin) -> bool:
    """Descarta soluções degeneradas (variância de um regime → 0): exige
    variância de cada regime ≥ 1% da variância do choque do AR(1) linear."""
    if not np.isfinite(r.llf):
        return False
    s2 = [v for n, v in par(r).items() if n.startswith("sigma2")]
    return min(s2) >= 0.01 * s2_lin


def ajustar_ms(y: pd.Series, k=2, tvtp: pd.Series | None = None, n_chutes=30,
               semente=1999, inicio=None):
    """MS com média e variância por regime e AR(1) comum. `tvtp`: variável
    (já defasada) que dirige as probabilidades de transição."""
    df = pd.DataFrame({"y": y, "y_lag": y.shift(1)})
    if tvtp is not None:
        df["z"] = tvtp
    df = df.dropna()
    kw = dict(k_regimes=k, trend="c", exog=df[["y_lag"]], switching_exog=False,
              switching_variance=True)
    if tvtp is not None:
        kw["exog_tvtp"] = sm.add_constant(df[["z"]])
    mod = MarkovRegression(df["y"], **kw)
    lin = sm.OLS(df["y"], sm.add_constant(df[["y_lag"]])).fit()
    rng = np.random.default_rng(semente)
    melhor = None
    for j in range(n_chutes):
        if j == 0 and inicio is not None:
            sp = inicio
        elif melhor is None:
            sp = None
        else:
            sp = np.asarray(melhor.params) + rng.normal(0, 0.5, len(melhor.params))
        try:
            r = mod.fit(start_params=sp, search_reps=20, disp=False)
        except Exception:
            continue
        if valido(r, lin.scale) and (melhor is None or r.llf > melhor.llf):
            melhor = r
    if melhor is None:
        raise RuntimeError("nenhuma solução não degenerada")
    return df, lin, melhor


def regime_alto(res) -> int:
    P = par(res)
    return int(np.argmax([P[f"const[{i}]"] for i in range(res.k_regimes)]))


def inicio_tvtp(res_fixo, n_z=1):
    """Ponto de partida do TVTP a partir do MS com probabilidades fixas."""
    P = np.asarray(res_fixo.params)
    logit = lambda q: np.log(q / (1 - q))
    return np.r_[logit(P[0]), logit(P[1]), np.zeros(2 * n_z), P[2:]]
