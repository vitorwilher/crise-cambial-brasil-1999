"""Índice de pressão cambial (EMP), Eichengreen, Rose e Wyplosz (1995) /
Kaminsky, Lizondo e Reinhart (1998).

    EMP_t = Δe/e − (σ_e/σ_r)·Δr/r + (σ_e/σ_i)·Δ(i − i*)

Δe/e: depreciação da PTAX (%); Δr/r: variação das reservas (%);
Δ(i − i*): variação do diferencial Selic − Fed funds (p.p.).
Os pesos igualam a volatilidade dos três componentes. Pressão "alta"/"crise":
EMP acima da média + 1,5σ / + 3σ.

Por que medir a pressão assim: sob câmbio administrado, um ataque pode ser
absorvido pelo câmbio (desvalorização), pelas reservas (o BC vende dólares) ou
pelos juros (o BC sobe a Selic). O Markov-switching de 30_markov_juros.py só enxerga
o canal dos juros; o EMP soma os três.

Duas versões:
  - mensal, jul/1994–dez/2000 (reservas mensais 3546); pesos e limiares
    calculados no período da banda (jul/1995–dez/1998), sem a flutuação;
  - semanal, set/1998–mar/1999 (reservas diárias 13621 só existem a partir de
    set/1998); pesos calculados na própria janela.

Cuidado: as reservas incluem os desembolsos do acordo com o FMI (a partir de
dez/1998), que sobem as reservas e ESCONDEM pressão no EMP.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DADOS, SAIDAS = RAIZ / "dados", RAIZ / "saidas"
sys.path.insert(0, str(RAIZ / "src"))
import estilo_am as am  # noqa: E402

d = pd.read_csv(DADOS / "painel_diario.csv", index_col=0, parse_dates=True)
res_m = pd.read_csv(DADOS / "reservas_mensal.csv", index_col=0, parse_dates=True)


def emp(e: pd.Series, r: pd.Series, i: pd.Series, janela: slice):
    comp = pd.DataFrame({
        "de": e.pct_change() * 100,
        "dr": r.pct_change() * 100,
        "di": i.diff(),
    })
    sd = comp.loc[janela].std()
    w_r, w_i = sd["de"] / sd["dr"], sd["de"] / sd["di"]
    comp["c_cambio"] = comp["de"]
    comp["c_reservas"] = -w_r * comp["dr"]
    comp["c_juros"] = w_i * comp["di"]
    comp["emp"] = comp[["c_cambio", "c_reservas", "c_juros"]].sum(axis=1, min_count=3)
    m, s = comp.loc[janela, "emp"].mean(), comp.loc[janela, "emp"].std()
    comp["z"] = (comp["emp"] - m) / s
    pesos = {"w_reservas": w_r, "w_juros": w_i, "media": m, "dp": s,
             "lim_1.5dp": m + 1.5 * s, "lim_3dp": m + 3 * s}
    return comp, pesos


# ---------------- mensal ----------------
idx_m = res_m.index  # 1º dia do mês, posição de fim de mês
e_m = d["ptax"].resample("MS").last().reindex(idx_m)
i_m = (d["selic"] - d["fedfunds"]).resample("MS").mean().reindex(idx_m)
r_m = res_m["reservas_mensal"]
mens, pesos_m = emp(e_m, r_m, i_m, slice("1995-07-01", "1998-12-01"))

# probabilidade do regime de ataque (Markov base) em média mensal
pm = pd.read_csv(SAIDAS / "markov_probabilidades.csv", index_col=0, parse_dates=True)
mens["prob_ataque_ms"] = pm["y_total fixa"].resample("MS").mean().reindex(idx_m)

mens.to_csv(SAIDAS / "emp_mensal.csv", index_label="mes")
print("Pesos (mensal, banda jul/95–dez/98):",
      {k: round(v, 3) for k, v in pesos_m.items()})
print("\nMeses acima de média + 1,5 dp:")
alto = mens[mens["z"] > 1.5]
print(alto[["c_cambio", "c_reservas", "c_juros", "emp", "z", "prob_ataque_ms"]].round(2))
print("\nCorrelação EMP × prob. de ataque (jul/95–dez/98):",
      round(mens.loc["1995-07":"1998-12", ["emp", "prob_ataque_ms"]].corr().iloc[0, 1], 2))

# ---------------- semanal (zoom) ----------------
dz = d.loc["1998-08-28":"1999-03-31"]
e_w = dz["ptax"].resample("W-FRI").last()
r_w = dz["reservas"].resample("W-FRI").last()
i_w = (dz["selic"] - dz["fedfunds"]).resample("W-FRI").mean()
sem, pesos_w = emp(e_w, r_w, i_w, slice("1998-09-04", "1999-03-26"))
sem["reservas"] = r_w
sem.to_csv(SAIDAS / "emp_semanal_zoom.csv", index_label="semana")
print("\nPesos (semanal, set/98–mar/99):", {k: round(v, 3) for k, v in pesos_w.items()})
print(sem[["reservas", "c_cambio", "c_reservas", "c_juros", "emp", "z"]].round(2).to_string())

# ---------------- gráficos ----------------
CORES = {"c_cambio": am.NAVY, "c_reservas": am.BLUE, "c_juros": am.AMBER}
NOMES = {"c_cambio": "Câmbio", "c_reservas": "Perda de reservas", "c_juros": "Juros"}


def barras(ax, df, largura):
    pos = np.zeros(len(df)); neg = np.zeros(len(df))
    for c in CORES:
        v = df[c].fillna(0).values
        base = np.where(v >= 0, pos, neg)
        ax.bar(df.index, v, bottom=base, width=largura, color=CORES[c], label=NOMES[c])
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.plot(df.index, df["emp"], color=am.RED, lw=1.2, marker="o", ms=2.5, label="EMP")
    ax.axhline(0, color=am.MUTED, lw=0.8)


fig, axs = am.figura(2, 1, figsize=(9.5, 7.5))
ax = axs[0]; am.eixo(ax)
mm = mens.loc["1995-01":"1999-03"]
barras(ax, mm, 22)
ax.axhline(pesos_m["lim_1.5dp"], color=am.RED, lw=0.8, ls=":")
ax.axhline(pesos_m["lim_3dp"], color=am.RED, lw=0.8, ls="--")
ax.text(pd.Timestamp("1996-01-01"), pesos_m["lim_3dp"], "média + 3 dp", va="bottom",
        fontsize=7, color=am.RED)
ax.text(pd.Timestamp("1996-09-01"), pesos_m["lim_1.5dp"], "média + 1,5 dp", va="bottom",
        fontsize=7, color=am.RED)
TOPO = 6.5
ax.set_ylim(-2, TOPO)
for dt, v in mm["emp"].items():
    if v > TOPO:
        ax.annotate(f"jan/99: {am.br(v)}", xy=(dt, TOPO), xytext=(-70, -12),
                    textcoords="offset points", fontsize=7, color=am.RED,
                    arrowprops=dict(arrowstyle="->", color=am.RED, lw=0.7))
am.subtitulo_eixo(ax, "Mensal, jan/1995–mar/1999 (pesos e limiares estimados na banda, jul/95–dez/98)")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
ax2 = ax.twinx()
ax2.plot(mm.index, mm["prob_ataque_ms"], color=am.GREEN, lw=1.2, ls="-")
ax2.set_ylim(0, 1.6); ax2.set_yticks([0, 0.5, 1])
ax2.tick_params(axis="y", colors=am.GREEN, labelsize=7)
for sp in ax2.spines.values():
    sp.set_visible(False)
ax2.text(mm.index[-1], 1.25, "prob. de ataque\n(Markov, eixo dir.)", color=am.GREEN,
         fontsize=7, ha="right", va="bottom")

ax = axs[1]; am.eixo(ax)
barras(ax, sem.dropna(subset=["emp"]), 5)
ax.set_ylim(-30, 42)
for dt, nome, ha in [("1998-11-13", "Acordo FMI ", "right"), ("1999-01-06", "Minas ", "right"),
                     ("1999-01-15", " Banda diagonal (13/1)\n e flutuação (15/1)", "left")]:
    ax.axvline(pd.Timestamp(dt), color=am.RED, lw=0.6, ls="--")
    ax.text(pd.Timestamp(dt), 41, nome, fontsize=7, color=am.RED, va="top", ha=ha)
ax.annotate("salto de reservas em 18/12\n(desembolso externo?)", xy=(pd.Timestamp("1998-12-18"), -27),
            xytext=(-120, 0), textcoords="offset points", fontsize=7, color=am.INK_SOFT,
            arrowprops=dict(arrowstyle="->", color=am.INK_SOFT, lw=0.7), va="center")
am.subtitulo_eixo(ax, "Semanal, set/1998–mar/1999 (reservas diárias; pesos na própria janela)")
am.titulos(fig, "Índice de pressão cambial: por onde o ataque passou",
           "EMP = depreciação − (σe/σr)·Δreservas% + (σe/σi)·Δ(Selic − Fed funds); "
           "contribuições em p.p.")
am.rodape(fig, "Fonte: BCB/SGS (PTAX 1, Selic 1178, reservas 3546 e 13621), FRED (DFF). "
          "Reservas incluem desembolsos do FMI. Elaboração própria.")
am.salvar(fig, SAIDAS / "emp_pressao_cambial.png")
