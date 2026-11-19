"""Brasil, 1999: por que a banda cambial caiu? — apresentação e simulador.

Aba 1: os slides do trabalho (HTML revealjs servido de www/).
Aba 2: simulador do contrafactual da dívida líquida do Gov. Federal + BC,
       a mesma conta de src/40_contrafactual.py:

    D_t = D_{t−1}·(1 + r_t) + P_t + A_t

com a base observada (BCB/SGS) em dados/contrafactual_base.csv.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from shiny import App, render, ui  # noqa: E402

AQUI = Path(__file__).parent
m = pd.read_csv(AQUI / "dados" / "contrafactual_base.csv", index_col=0, parse_dates=True)
PAR = json.load(open(AQUI / "dados" / "parametros.json"))
B95, B98, REF, S99 = PAR["b95"], PAR["b98"], PAR["selic_ref"], PAR["s99"]

NAVY, BLUE, AMBER, RED, MUTED = "#0A1A3C", "#1CA0D8", "#F5A524", "#E5484D", "#8894A5"
DEFESA = (m.index >= "1997-11-01") & (m.index <= "1998-12-01")
EM98 = (m.index >= "1998-01-01") & (m.index <= "1998-12-01")


def simular(superavit=None, inicio="1996-01", sens98=B98, teto_selic=None):
    r_aa = ((1 + m["r"]) ** 12 - 1) * 100
    desvio = m["selic"] - REF
    delta = pd.Series(0.0, index=m.index)
    delta[EM98] += ((B98 - sens98) * desvio)[EM98]
    if teto_selic is not None:
        corte = (m["selic"] - np.minimum(m["selic"], teto_selic))
        delta[DEFESA] += (sens98 * corte)[DEFESA]
    r = (1 + (r_aa - delta) / 100) ** (1 / 12) - 1
    P = m["P"].copy()
    if superavit is not None:
        desde = m.index >= pd.Timestamp(inicio + "-01")
        P[desde] = -superavit / 100 * m.loc[desde, "pib"]
    D, prev = [], m["D0"].iloc[0]
    for t in range(len(m)):
        prev = prev * (1 + r.iloc[t]) + P.iloc[t] + m["A"].iloc[t]
        D.append(prev)
    D = pd.Series(D, index=m.index)
    return D / m["pibv"] * 100


OBS = simular()

texto_metodo = ui.markdown(f"""
**Como funciona.** A simulação reproduz a dívida líquida federal observada mês a mês
(1996–1999) e troca só o que você mudar. Selic, câmbio, crescimento e ajustes
patrimoniais ficam como foram (equilíbrio parcial).

- **Primário:** superávit em % do PIB a partir da data escolhida (em 1999 foi {S99:.1f}%).
- **Sensibilidade à Selic em 1998:** {B95:.2f} era a de 1995–97 (dívida mais prefixada);
  {B98:.2f} foi a de 1998 (dívida 69% na Selic).
- **Teto da Selic na defesa (nov/97–dez/98):** limita o juro usado contra os ataques
  da Ásia e da Rússia; a Selic média antes da Ásia era {REF:.1f}%.

Fonte: BCB/SGS (4468, 4606, 4639, 4380, 190, 1178). Código:
[github.com/vitorwilher/crise-cambial-brasil-1999](https://github.com/vitorwilher/crise-cambial-brasil-1999).
""")

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Apresentação",
        ui.tags.iframe(src="www/slides.html", style="width:100%; height:82vh; border:0;"),
    ),
    ui.nav_panel(
        "Simulador do contrafactual",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_switch("usa_primario", "Mudar o primário", value=True),
                ui.input_slider("primario", "Superávit primário (% do PIB)",
                                min=-1, max=4, value=round(S99, 1), step=0.1),
                ui.input_select("inicio", "A partir de",
                                {"1996-01": "jan/1996", "1997-01": "jan/1997",
                                 "1997-11": "nov/1997 (Ásia)", "1998-09": "set/1998 (Rússia)"}),
                ui.input_slider("sens", "Sensibilidade do custo da dívida à Selic em 1998",
                                min=0.0, max=0.6, value=round(B98, 2), step=0.01),
                ui.input_switch("usa_teto", "Limitar a Selic na defesa", value=False),
                ui.input_slider("teto", "Teto da Selic (% a.a.)", min=15, max=45,
                                value=round(REF), step=1),
                width=340,
            ),
            ui.output_plot("grafico", height="430px"),
            ui.output_ui("resumo"),
            texto_metodo,
        ),
    ),
    title="Brasil, 1999: por que a banda cambial caiu?",
    id="abas",
)


def server(input, output, session):
    def cenario():
        return simular(
            superavit=input.primario() if input.usa_primario() else None,
            inicio=input.inicio(), sens98=input.sens(),
            teto_selic=input.teto() if input.usa_teto() else None,
        )

    @render.plot
    def grafico():
        cf = cenario()
        fig, ax = plt.subplots(figsize=(9, 4.3))
        ax.plot(OBS.index, OBS, color=NAVY, lw=2.4, label="Observado")
        ax.plot(cf.index, cf, color=RED, lw=1.8, ls="--", label="Seu cenário")
        for dt, nome in {"1997-10-23": "Ásia", "1998-08-17": "Rússia",
                         "1999-01-15": "Flutuação"}.items():
            ax.axvline(pd.Timestamp(dt), color=MUTED, lw=0.7, ls=":")
            ax.text(pd.Timestamp(dt), ax.get_ylim()[1], " " + nome, fontsize=8,
                    color=MUTED, va="top")
        ax.set_title("Dívida líquida do Governo Federal + BC (% do PIB)", loc="left",
                     fontsize=11, color=NAVY)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#E6EAF0")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        return fig

    @render.ui
    def resumo():
        cf = cenario()
        linhas = []
        for rot, dt in [("dez/1997", "1997-12-01"), ("dez/1998", "1998-12-01"),
                        ("dez/1999", "1999-12-01")]:
            o, c = OBS.loc[dt], cf.loc[dt]
            linhas.append(f"<tr><td>{rot}</td><td>{o:.1f}</td><td>{c:.1f}</td>"
                          f"<td>{c - o:+.1f}</td></tr>")
        return ui.HTML(
            "<table class='table table-sm' style='max-width:520px'>"
            "<thead><tr><th></th><th>Observado</th><th>Seu cenário</th>"
            "<th>Diferença (p.p.)</th></tr></thead><tbody>" + "".join(linhas) +
            "</tbody></table>")


app = App(app_ui, server, static_assets={"/www": AQUI / "www"})
