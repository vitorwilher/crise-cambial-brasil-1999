"""Prazo e composição da dívida mobiliária como sinal e motor do regime de ataque.

Hipótese: o encurtamento do financiamento sinaliza piora das condições de
mercado (Broner, Lorenzoni e Schmukler, 2013) e amplia o risco de rolagem
(Cole e Kehoe, 2000). Testes:

1. TVTP: o prazo médio de emissão das LTN no mês anterior (SGS 10613) muda a
   probabilidade de entrar/sair do regime de ataque do MS-AR(1) de 30_markov_juros.py?
   Amostra: set/1996 (início da série) a 08/01/1999. Comparação com o MS de
   probabilidades fixas na MESMA amostra.
2. Antecipação: o prazo cai ANTES da entrada no regime de ataque? Médias do
   prazo nos meses anteriores a cada entrada.
3. Composição anual por indexador (Relatórios Anuais do BC, transcrita em
   referencias/fontes_divida/composicao_indexador_1994_1999.csv).

Cuidado: o prazo é endógeno (responde à mesma piora). Usamos sempre o valor
do mês anterior e interpretamos como indicador antecedente, não como choque.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ms_util import ajustar_ms, inicio_tvtp, par, pv, regime_alto  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DADOS, SAIDAS = RAIZ / "dados", RAIZ / "saidas"
sys.path.insert(0, str(RAIZ / "src"))
import estilo_am as am  # noqa: E402

INICIO, FIM = "1996-09-06", "1999-01-08"

s = pd.read_csv(DADOS / "painel_semanal.csv", index_col=0, parse_dates=True)
y = (s["dif_juros"] - s["deslize"]).loc[INICIO:FIM]
z = s["prazo_ltn"].loc[INICIO:FIM]   # já é o valor do mês anterior

# ---- 1. MS fixo × TVTP com prazo ----
df0, lin, fixo = ajustar_ms(y)
df1, _, tv = ajustar_ms(y, tvtp=z, inicio=inicio_tvtp(fixo))
k0, k1 = regime_alto(fixo), regime_alto(tv)
P1, PV1 = par(tv), pv(tv)

linhas = []
for nome, r, k in [("MS prob. fixa", fixo, k0), ("MS TVTP prazo LTN", tv, k1)]:
    P = par(r)
    rho = P[[n for n in P.index if n.startswith("x1")][0]]
    linhas.append({
        "modelo": nome, "n": int(r.nobs), "llf": round(r.llf, 2),
        "aic": round(r.aic, 1), "bic": round(r.bic, 1),
        "media_calmo": round(P[f"const[{1-k}]"] / (1 - rho), 1),
        "media_ataque": round(P[f"const[{k}]"] / (1 - rho), 1),
        "regime_ataque": k,
    })
tab = pd.DataFrame(linhas).set_index("modelo")
lr = 2 * (tv.llf - fixo.llf)
coef = {n: f"{P1[n]:.3f} (p={PV1[n]:.3f})" for n in P1.index if "tvtp" in n}
print(tab.T)
print("\nCoeficientes TVTP (logit de p[i->0]; regime de ataque =", k1, "):")
for n, v in coef.items():
    print(f"  {n}: {v}")
print(f"LR (TVTP × fixo, 2 g.l.): {lr:.2f}")

# Leitura: p[i->0] é a prob. de ir para o regime 0. Converte para efeitos sobre
# a probabilidade de ENTRAR no ataque (calmo→ataque) e de PERMANECER no ataque.
pm = tv.regime_transition[:, :, :] if tv.regime_transition.ndim == 3 else None
trans = pd.DataFrame({
    "p_entrar_ataque": tv.regime_transition[k1, 1 - k1, :],
    "p_ficar_ataque": tv.regime_transition[k1, k1, :],
    "prazo_ltn": df1["z"].values,
}, index=df1.index)
trans.to_csv(SAIDAS / "prazo_tvtp_transicoes.csv", index_label="semana")
print("\nProb. de entrar no ataque por faixa de prazo (meses):")
print(trans.groupby(pd.cut(trans["prazo_ltn"], [0, 3, 4, 6]))
      [["p_entrar_ataque", "p_ficar_ataque"]].mean().round(3))

# ---- 2. O prazo cai antes da entrada no ataque? ----
p_fixo = pd.Series(np.asarray(fixo.smoothed_marginal_probabilities[k0]), index=df0.index)
ataque = (p_fixo > 0.5).astype(int)
entradas = ataque.index[(ataque.diff() == 1)]
pz_m = pd.read_csv(DADOS / "prazo_ltn.csv", index_col=0, parse_dates=True)["prazo_ltn"]
print("\nEntradas no regime de ataque e prazo das LTN emitidas nos meses anteriores:")
for e in entradas:
    m = e.to_period("M").to_timestamp()
    jan = pz_m.loc[m - pd.DateOffset(months=6): m]
    print(f"  {e.date()}: " + ", ".join(f"{d:%b/%y} {v:.2f}" for d, v in jan.items()))

resumo = pd.concat([tab, pd.DataFrame({"LR_tvtp_vs_fixo": [np.nan, round(lr, 2)]},
                                      index=tab.index)], axis=1)
for n, v in coef.items():
    resumo.loc["MS TVTP prazo LTN", n] = v
resumo.to_csv(SAIDAS / "prazo_markov_resultados.csv")

# ---- 3. gráfico ----
comp = pd.read_csv(RAIZ / "referencias" / "fontes_divida" /
                   "composicao_indexador_1994_1999.csv", comment="#")
fig, axs = am.figura(1, 2, figsize=(13, 5.4))
ax = axs[0]; am.eixo(ax)
pz_plot = pz_m.loc["1996-08":"1999-12"]
ax.plot(pz_plot.index, pz_plot.values, color=am.NAVY, lw=1.8, marker="o", ms=3,
        label="LTN prefixadas (SGS 10613)")
pl = pd.read_csv(DADOS / "prazo_lft.csv", index_col=0, parse_dates=True)["prazo_lft"]
pl = pl.loc[:"1999-12"]
ax.plot(pl.index, pl.values, color=am.AMBER, lw=1.4, marker="o", ms=2.5,
        label="LFT, Selic (SGS 10614)")
ax2 = ax.twinx()
ax2.fill_between(p_fixo.index, p_fixo.values, color=am.BLUE_WASH, alpha=0.6, step="mid")
ax2.set_ylim(0, 1.05); ax2.set_yticks([0, 0.5, 1])
ax2.tick_params(axis="y", colors=am.BLUE, labelsize=7)
for sp in ax2.spines.values():
    sp.set_visible(False)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
for dt, nome in {"1997-10-23": "Hong Kong", "1998-08-17": "Rússia",
                 "1999-01-15": "Flutuação"}.items():
    ax.axvline(pd.Timestamp(dt), color=am.RED, lw=0.7, ls="--")
    ax.text(pd.Timestamp(dt), ax.get_ylim()[1], nome + " ", fontsize=7, color=am.RED,
            rotation=90, va="top", ha="right")
am.subtitulo_eixo(ax, "Prazo médio de emissão (meses)")
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0, 0.72))

ax = axs[1]; am.eixo(ax)
grupos = {"Prefixado": ("prefixado", am.NAVY), "Over/Selic": ("selic", am.AMBER),
          "Câmbio": ("cambio", am.BLUE), "TR e outros": (None, am.MUTED)}
outros = comp[["tr", "igpm", "tjlp", "tbf", "igpdi"]].fillna(0).sum(axis=1)
base = np.zeros(len(comp))
for nome, (col, cor) in grupos.items():
    v = (comp[col] if col else outros).fillna(0).values
    ax.bar(comp["ano"], v, bottom=base, color=cor, label=nome, width=0.65)
    for xi, b, vi in zip(comp["ano"], base, v):
        if vi >= 6:
            ax.text(xi, b + vi / 2, am.br(vi), ha="center", va="center",
                    fontsize=7, color="white")
    base += v
ax.set_ylim(0, 100)
am.subtitulo_eixo(ax, "Títulos federais por indexador, dezembro (% do total)")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, -0.06))
am.titulos(fig, "O encurtamento e a indexação da dívida como sinal do ataque",
           "Com a crise, o Tesouro troca LTN prefixada e longa por LFT e títulos cambiais")
am.rodape(fig, "Fonte: BCB/SGS (10613, 10614); BCB, Relatórios Anuais 1997–1999 "
          "(Quadros 3.5, 4.5, 4.8). Elaboração própria.")
am.eixo_datas(axs[0])
ax2.text(0.01, 0.6, "área azul: prob. do regime\nde ataque (eixo dir.)", transform=axs[0].transAxes,
         ha="left", va="top", fontsize=8, color=am.BLUE,
         bbox=dict(facecolor=am.PAPER, edgecolor="none", pad=1))
fig.subplots_adjust(top=0.82, wspace=0.18, bottom=0.14)
am.salvar(fig, SAIDAS / "prazo_composicao_divida.png")
