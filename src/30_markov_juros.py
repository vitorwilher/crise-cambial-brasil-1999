"""Markov-switching na expectativa de desvalorização (Jeanne e Masson, 2000).

Pergunta: o regime de bandas (jul/1995 a jan/1999) alternou entre um equilíbrio
"calmo" e um "de ataque"? Quando a economia entrou no regime ruim, e o que
(reservas) move a probabilidade de troca?

Variáveis dependentes (p.p. ao ano, semanal):
  y_total  = Selic − Fed funds − deslize   (risco-país + prêmio cambial)
  y_cambio = y_total − EMBI/100            (só o prêmio cambial)

Modelos:
  M0  linear AR(1)                             (referência)
  M1  MS(2): média e variância mudam, AR(1) comum (Hamilton, 1989)
  M2  M1 com probabilidades de transição variáveis (TVTP), dirigidas pela
      variação das reservas em 3 meses (Diebold, Lee e Weinbach, 1994)
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
INICIO, FIM = "1995-07-07", "1999-01-08"   # minibandas até a véspera da banda diagonal

s = pd.read_csv(DADOS / "painel_semanal.csv", index_col=0, parse_dates=True)
s["y_total"] = s["dif_juros"] - s["deslize"]
s["y_cambio"] = s["premio_cambial"]


def ajustar(nome_y: str, tvtp: bool = False):
    df = s[[nome_y, "var_reservas_3m"]].copy()
    df["y_lag"] = df[nome_y].shift(1)
    df = df.loc[INICIO:FIM].dropna()
    y, x = df[nome_y], df[["y_lag"]]

    lin = sm.OLS(y, sm.add_constant(x)).fit()

    kw = dict(k_regimes=2, trend="c", exog=x, switching_exog=False,
              switching_variance=True)
    if tvtp:
        kw["exog_tvtp"] = sm.add_constant(df[["var_reservas_3m"]].shift(1).bfill())
    mod = MarkovRegression(y, **kw)
    melhor = None
    rng = np.random.default_rng(1999)
    for _ in range(30):   # vários chutes iniciais: a verossimilhança é multimodal
        try:
            r = mod.fit(start_params=None if melhor is None else
                        np.asarray(melhor.params) + rng.normal(0, 0.5, len(melhor.params)),
                        search_reps=20, disp=False)
        except Exception:
            continue
        if melhor is None or r.llf > melhor.llf:
            melhor = r
    return df, lin, melhor


def par(res) -> pd.Series:
    """Parâmetros com nome (o statsmodels chama o AR de 'x1')."""
    return pd.Series(np.asarray(res.params), index=res.model.param_names)


def pv(res) -> pd.Series:
    return pd.Series(np.asarray(res.pvalues), index=res.model.param_names)


def regime_ruim(res) -> int:
    """Regime de média de longo prazo mais alta = 'ataque'."""
    c = [par(res)[f"const[{i}]"] for i in range(2)]
    return int(np.argmax(c))


linhas, probs = [], {}
for y in ["y_total", "y_cambio"]:
    for tvtp in [False, True]:
        df, lin, res = ajustar(y, tvtp)
        k = regime_ruim(res)
        P = par(res)
        nome_ar = [n for n in P.index if n.startswith("x1")][0]
        rho = P[nome_ar]
        rotulo = f"{y} {'TVTP' if tvtp else 'fixa'}"
        p = res.smoothed_marginal_probabilities[k]
        probs[rotulo] = p
        lin_ll = lin.llf
        linha = {
            "modelo": rotulo, "n": int(res.nobs),
            "llf_linear": round(lin_ll, 1), "llf_ms": round(res.llf, 1),
            "LR": round(2 * (res.llf - lin_ll), 1),
            "aic_linear": round(lin.aic, 1), "aic_ms": round(res.aic, 1),
            "rho": round(rho, 3),
            "media_calmo": round(P[f"const[{1-k}]"] / (1 - rho), 1),
            "media_ataque": round(P[f"const[{k}]"] / (1 - rho), 1),
            "dp_calmo": round(np.sqrt(P[f"sigma2[{1-k}]"]), 2),
            "dp_ataque": round(np.sqrt(P[f"sigma2[{k}]"]), 2),
            "semanas_ataque": int((p > 0.5).sum()),
        }
        if not tvtp:
            dur = res.expected_durations
            linha["dur_calmo"] = round(dur[1 - k], 1)
            linha["dur_ataque"] = round(dur[k], 1)
        else:
            PV = pv(res)
            for nome, v in P.items():
                if nome.startswith("p[") and "const" not in nome:
                    linha[nome] = f"{v:.3f} (p={PV[nome]:.3f})"
        linhas.append(linha)
        print(f"\n===== {rotulo} =====")
        print(res.summary().tables[1:])

tab = pd.DataFrame(linhas).set_index("modelo")
tab.to_csv(SAIDAS / "markov_resultados.csv")
pd.DataFrame(probs).to_csv(SAIDAS / "markov_probabilidades.csv", index_label="semana")
pd.set_option("display.width", 250)
print("\n", tab.T)

# Semanas em regime de ataque (modelo base: y_total, prob. fixa)
p = probs["y_total fixa"]
ataque = p[p > 0.5]
blocos = (ataque.index.to_series().diff() > pd.Timedelta(days=7)).cumsum()
print("\nEpisódios em regime de ataque (y_total, prob. fixa):")
for _, g in ataque.groupby(blocos.values):
    print(f"  {g.index[0].date()} a {g.index[-1].date()}  ({len(g)} semanas)")

# ---------- gráfico ----------
EVENTOS = {"1997-10-23": "Hong Kong", "1998-08-17": "Rússia",
           "1998-10-04": "Eleição", "1998-11-13": "FMI",
           "1999-01-06": "Minas"}
fig, axs = am.figura(1, 2, figsize=(13, 5.2), sharex=True)
ax = axs[0]
am.eixo(ax)
d = s.loc[INICIO:FIM]
ax.plot(d.index, d["y_total"], color=am.NAVY, lw=1.6, label="Selic − Fed funds − deslize")
ax.plot(d.index, d["y_cambio"], color=am.BLUE, lw=1.3, label="… − EMBI (prêmio cambial)")
ax.axhline(0, color=am.MUTED, lw=0.8)
am.subtitulo_eixo(ax, "Prêmio implícito nos juros (p.p. a.a.)")
ax.legend(frameon=False, fontsize=8, loc="upper center")
ax = axs[1]
am.eixo(ax)
ax.fill_between(probs["y_total fixa"].index, probs["y_total fixa"], color=am.BLUE_WASH,
                step="mid", label="y_total, prob. fixa")
ax.plot(probs["y_total TVTP"].index, probs["y_total TVTP"], color=am.NAVY, lw=1.2,
        label="y_total, TVTP (reservas)")
ax.plot(probs["y_cambio fixa"].index, probs["y_cambio fixa"], color=am.AMBER, lw=1.2,
        label="prêmio cambial, prob. fixa")
ax.set_ylim(0, 1.05)
am.subtitulo_eixo(ax, "Probabilidade suavizada do regime de ataque")
ax.legend(frameon=False, fontsize=8, loc="center")
for a in axs:
    for dt, nome in EVENTOS.items():
        a.axvline(pd.Timestamp(dt), color=am.RED, lw=0.7, ls="--")
for dt, nome in EVENTOS.items():
    axs[0].text(pd.Timestamp(dt), axs[0].get_ylim()[1], nome, rotation=90,
                va="top", ha="right", fontsize=7, color=am.RED)
am.titulos(fig, "Brasil, 1995–1999: dois regimes nos juros da banda cambial",
           "Markov-switching com 2 regimes, AR(1), semanal, jul/1995 a 08/01/1999")
am.rodape(fig, "Fonte: BCB/SGS (PTAX 1, Selic 1178, reservas 3546), FRED (DFF), "
          "Ipeadata (EMBI+ JPM366_EMBI366). Elaboração própria.")
for a in axs:
    am.eixo_datas(a)
fig.subplots_adjust(top=0.82, wspace=0.18)
am.salvar(fig, SAIDAS / "markov_probabilidades.png")
