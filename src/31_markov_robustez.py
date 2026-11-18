"""Robustez do Markov-switching (modelo base: 30_markov_juros.py, y_total semanal).

1. Teste de linearidade por bootstrap paramétrico. Sob H0 (1 regime) as
   probabilidades de transição não são identificadas e a LR não tem
   distribuição qui-quadrado (Hansen, 1992; Garcia, 1998). Simulamos o AR(1)
   linear estimado, reestimamos linear e MS em cada réplica e comparamos a LR
   observada com a distribuição simulada.
2. Especificações alternativas:
   - frequência diária (deslize em 63 dias úteis);
   - 3 regimes;
   - amostras começando em jan/1996 e jan/1997 (tira a desinflação pós-Real);
   - deslize medido em 26 semanas;
   - probabilidade de transição dirigida pelo EMBI defasado (TVTP).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

RAIZ = Path(__file__).resolve().parents[1]
DADOS, SAIDAS = RAIZ / "dados", RAIZ / "saidas"
sys.path.insert(0, str(RAIZ / "src"))
import estilo_am as am  # noqa: E402

warnings.filterwarnings("ignore")
FIM = "1999-01-08"
N_BOOT = 499

s = pd.read_csv(DADOS / "painel_semanal.csv", index_col=0, parse_dates=True)
d = pd.read_csv(DADOS / "painel_diario.csv", index_col=0, parse_dates=True)


def deslize(ptax: pd.Series, janela: int, por_ano: int) -> pd.Series:
    return ((ptax / ptax.shift(janela)) ** (por_ano / janela) - 1) * 100


def valido(r, s2_lin) -> bool:
    """Descarta soluções degeneradas: a verossimilhança da mistura é ilimitada
    quando a variância de um regime vai a zero (um regime 'captura' poucas
    observações). Exige variância de cada regime ≥ 1% da variância do choque
    do AR(1) linear."""
    if not np.isfinite(r.llf):
        return False
    nomes = r.model.param_names
    s2 = [v for n, v in zip(nomes, np.asarray(r.params)) if n.startswith("sigma2")]
    return min(s2) >= 0.01 * s2_lin


def ajustar_ms(y: pd.Series, k=2, tvtp: pd.Series | None = None, n_chutes=30,
               semente=1999, inicio=None):
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
        try:
            if j == 0 and inicio is not None:
                sp = inicio
            elif melhor is None:
                sp = None
            else:
                sp = np.asarray(melhor.params) + rng.normal(0, 0.5, len(melhor.params))
            r = mod.fit(start_params=sp, search_reps=20, disp=False)
        except Exception:
            continue
        if valido(r, lin.scale) and (melhor is None or r.llf > melhor.llf):
            melhor = r
    if melhor is None:
        raise RuntimeError("nenhuma solução não degenerada")
    return df, lin, melhor


def resumo(rotulo, df, lin, res):
    P = pd.Series(np.asarray(res.params), index=res.model.param_names)
    k = res.k_regimes
    rho = P[[n for n in P.index if n.startswith("x1")][0]]
    medias = np.array([P[f"const[{i}]"] for i in range(k)]) / (1 - rho)
    ordem = np.argsort(medias)
    alto = ordem[-1]
    p_alto = res.smoothed_marginal_probabilities[alto]
    p_alto = pd.Series(np.asarray(p_alto), index=df.index)
    linha = {
        "especificação": rotulo, "n": int(res.nobs), "k": k,
        "llf_linear": round(lin.llf, 1), "llf_ms": round(res.llf, 1),
        "aic_linear": round(lin.aic, 1), "aic_ms": round(res.aic, 1),
        "bic_linear": round(lin.bic, 1), "bic_ms": round(res.bic, 1),
        "rho": round(rho, 3),
        "regime_ataque (índice)": int(alto),
        "médias por regime": " / ".join(f"{m:.1f}" for m in medias[ordem]),
        "dp por regime": " / ".join(
            f"{np.sqrt(P[f'sigma2[{i}]']):.2f}" for i in ordem),
        "prob_ataque_dez98": round(p_alto.loc["1998-12-01":"1998-12-31"].mean(), 3),
        "prob_ataque_jan99": round(p_alto.loc["1999-01-01":FIM].mean(), 3),
    }
    if res.model.tvtp:
        PV = pd.Series(np.asarray(res.pvalues), index=res.model.param_names)
        for n in P.index:
            if n.endswith("tvtp1"):
                linha[n] = f"{P[n]:.3f} (p={PV[n]:.3f})"
    return linha, p_alto


linhas, probs = [], {}
y_sem = s["dif_juros"] - s["deslize"]

# --- base ---
df, lin, res = ajustar_ms(y_sem.loc["1995-07-07":FIM])
l, p = resumo("Base: semanal, 2 regimes, jul/95", df, lin, res)
linhas.append(l); probs["Base (semanal)"] = p
lr_obs = 2 * (res.llf - lin.llf)

# --- diária ---
yd = d["selic"] - d["fedfunds"] - deslize(d["ptax"], 63, 252)
df, lin, r = ajustar_ms(yd.loc["1995-07-07":FIM], n_chutes=15)
l, p = resumo("Diária, 2 regimes", df, lin, r)
linhas.append(l); probs["Diária"] = p

# --- 3 regimes ---
df, lin, r = ajustar_ms(y_sem.loc["1995-07-07":FIM], k=3, n_chutes=60)
l, p = resumo("Semanal, 3 regimes (regime mais alto)", df, lin, r)
linhas.append(l); probs["3 regimes"] = p

# --- amostras ---
for ini, nome in [("1996-01-05", "jan/96"), ("1997-01-03", "jan/97")]:
    df, lin, r = ajustar_ms(y_sem.loc[ini:FIM])
    l, p = resumo(f"Amostra desde {nome}", df, lin, r)
    linhas.append(l); probs[f"Desde {nome}"] = p

# --- deslize em 26 semanas ---
y26 = s["dif_juros"] - deslize(s["ptax"], 26, 52)
df, lin, r = ajustar_ms(y26.loc["1995-07-07":FIM])
l, p = resumo("Deslize em 26 semanas", df, lin, r)
linhas.append(l); probs["Deslize 26 sem."] = p

# --- TVTP com EMBI defasado (centenas de p.b.) ---
Pb = np.asarray(res.params)  # base: p00, p10, const0, const1, ar, s2_0, s2_1
logit = lambda q: np.log(q / (1 - q))
inicio = np.r_[logit(Pb[0]), logit(Pb[1]), 0.0, 0.0, Pb[2:]]
df, lin, r = ajustar_ms(y_sem.loc["1995-07-07":FIM],
                        tvtp=(s["embi"] / 100).shift(1).loc["1995-07-07":FIM],
                        inicio=inicio)
l, p = resumo("TVTP: EMBI defasado", df, lin, r)
linhas.append(l); probs["TVTP EMBI"] = p

tab = pd.DataFrame(linhas).set_index("especificação")

# --- bootstrap da LR ---
print(f"Bootstrap da LR com {N_BOOT} réplicas...")
yb = y_sem.loc["1995-07-07":FIM]
dfb = pd.DataFrame({"y": yb, "y_lag": yb.shift(1)}).dropna()
c, phi = lin_base = sm.OLS(dfb["y"], sm.add_constant(dfb["y_lag"])).fit().params.values
sig = np.sqrt(sm.OLS(dfb["y"], sm.add_constant(dfb["y_lag"])).fit().scale)
rng = np.random.default_rng(2026)
lrs = []
for b in range(N_BOOT):
    e = rng.normal(0, sig, len(dfb) + 1)
    ys = np.empty(len(dfb) + 1)
    ys[0] = dfb["y_lag"].iloc[0]
    for t in range(1, len(ys)):
        ys[t] = c + phi * ys[t - 1] + e[t]
    X = sm.add_constant(ys[:-1])
    ll0 = sm.OLS(ys[1:], X).fit().llf
    best = -np.inf
    mb = MarkovRegression(ys[1:], k_regimes=2, exog=ys[:-1, None],
                          switching_exog=False, switching_variance=True)
    for rep in range(8):
        try:
            rr = mb.fit(search_reps=20, disp=False)
        except Exception:
            continue
        if valido(rr, sm.OLS(ys[1:], X).fit().scale):
            best = max(best, rr.llf)
    if np.isfinite(best):
        # o MS aninha o linear: LR < 0 é falha de otimização, tratada como 0
        lrs.append(max(0.0, 2 * (best - ll0)))
lrs = np.array(lrs)
pd.Series(lrs, name="LR").to_csv(SAIDAS / "robustez_bootstrap_lr_replicas.csv", index=False)
p_boot = (1 + (lrs >= lr_obs).sum()) / (1 + len(lrs))
boot = pd.Series({
    "LR observada": round(lr_obs, 1), "réplicas válidas": len(lrs),
    "LR crítica 95%": round(np.quantile(lrs, 0.95), 1),
    "LR crítica 99%": round(np.quantile(lrs, 0.99), 1),
    "LR máxima simulada": round(lrs.max(), 1), "p-valor bootstrap": round(p_boot, 4),
})
boot.to_csv(SAIDAS / "robustez_bootstrap_lr.csv", header=["valor"])
tab.to_csv(SAIDAS / "robustez_markov.csv")
pd.set_option("display.width", 250, "display.max_colwidth", 40)
print(tab.T)
print(boot)

# --- gráfico ---
fig, ax = am.figura(1, 1, figsize=(9.5, 5.2))
am.eixo(ax)
cores = {"Base (semanal)": am.NAVY, "Diária": am.BLUE, "3 regimes": am.AMBER,
         "Desde jan/96": am.GREEN, "Desde jan/97": am.RED,
         "Deslize 26 sem.": am.MUTED, "TVTP EMBI": am.BLUE_DEEP}
for nome, p in probs.items():
    ax.plot(p.index, p, lw=2.2 if nome.startswith("Base") else 1.0,
            color=cores[nome], label=nome, alpha=1 if nome.startswith("Base") else 0.85)
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8, loc="center", ncol=2)
am.titulos(fig, "Robustez: probabilidade do regime de juros mais alto",
           "Probabilidades suavizadas, especificações alternativas, até 08/01/1999")
am.rodape(fig, "Fonte: BCB/SGS, FRED, Ipeadata. Elaboração própria.")
am.salvar(fig, SAIDAS / "robustez_probabilidades.png")
