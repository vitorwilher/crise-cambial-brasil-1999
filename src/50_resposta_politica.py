"""Bloco 5 do roteiro: resposta de política econômica, papel do FMI e custos.

1. O pacote do FMI comprou tempo, não credibilidade? Reservas diárias com os
   desembolsos documentados (FMI, BIS) e a perda que continuou até a flutuação.
2. A virada de regime: câmbio, Selic e EMBI de set/1998 a dez/1999.
3. O ajuste fiscal prometido e o entregue: primário federal (12 meses) contra
   as metas dos memorandos com o FMI.
4. A nova âncora: IPCA em 12 meses contra a meta de 1999 (8% ± 2 p.p.).
5. Custos: crescimento, inflação e o salto da dívida com a desvalorização.

Datas e números de política: referencias/fontes_fmi/cronologia_politica.csv,
com a fonte documental de cada linha.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DADOS, SAIDAS = RAIZ / "dados", RAIZ / "saidas"
sys.path.insert(0, str(RAIZ / "src"))
import estilo_am as am  # noqa: E402

d = pd.read_csv(DADOS / "painel_diario.csv", index_col=0, parse_dates=True)
fis = pd.read_csv(SAIDAS / "fiscal_painel.csv", index_col=0, parse_dates=True)
ipca = pd.read_csv(DADOS / "ipca_12m.csv", index_col=0, parse_dates=True)["ipca_12m"]
pib = pd.read_csv(DADOS / "pib_crescimento_anual.csv", index_col=0,
                  parse_dates=True)["pib_crescimento_anual"]
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def eixo_mes(a, meses):
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=meses))
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: (lambda t: f"{MESES[t.month - 1]}/{t:%y}")(mdates.num2date(x))))


crono = pd.read_csv(RAIZ / "referencias" / "fontes_fmi" / "cronologia_politica.csv",
                    comment="#", parse_dates=["data"])

z = d.loc["1998-09-01":"1999-12-31"]

# ---------- 1. reservas e desembolsos ----------
saltos = d.loc["1998-12-01":"1998-12-31", "reservas"].diff().nlargest(2)
res_pre_rus = d.loc["1998-09-01", "reservas"]
res_14dez = d.loc["1998-12-14", "reservas"]
res_12jan = d.loc["1999-01-12", "reservas"]
res_min = d.loc["1999-01-13":"1999-12-31", "reservas"]
print("Maiores altas diárias de reservas em dez/1998 (US$ mi):")
print(saltos)
print(f"Reservas: 01/09/98 {res_pre_rus:,.0f} → 14/12/98 {res_14dez:,.0f} → "
      f"12/01/99 {res_12jan:,.0f}; mínimo em 1999 {res_min.min():,.0f} "
      f"em {res_min.idxmin().date()}")
perda_pos_fmi = d.loc["1998-12-21", "reservas"] - res_12jan
print(f"Perda entre 21/12/98 (depois dos saques) e 12/01/99: {perda_pos_fmi:,.0f}")

# ---------- 2. picos e reversão ----------
def pico(col, ini="1999-01-13", fim="1999-12-31", maior=True):
    s = d.loc[ini:fim, col]
    return (s.max(), s.idxmax().date()) if maior else (s.min(), s.idxmin().date())


ptax_pico = pico("ptax")
selic_pico = pico("selic")
embi_pico = pico("embi", ini="1998-09-01")
print(f"PTAX pico {ptax_pico}; dez/99 média {d.loc['1999-12', 'ptax'].mean():.3f}")
print(f"Selic pico {selic_pico}; dez/99 média {d.loc['1999-12', 'selic'].mean():.1f}")
print(f"EMBI pico {embi_pico}; dez/99 média {d.loc['1999-12', 'embi'].mean():.0f}")

# ---------- 3. fiscal: meta × realizado ----------
prim_fed = -fis["primario_gf_12m_pib"]
print(f"Primário GF+BC 12m: dez/98 {prim_fed.loc['1998-12-01']:.2f}; "
      f"dez/99 {prim_fed.loc['1999-12-01']:.2f}; dez/00 {prim_fed.loc['2000-12-01']:.2f}")

# ---------- 5. custos ----------
custos = pd.DataFrame({
    "indicador": [
        "PIB, crescimento real 1998 (%)", "PIB, crescimento real 1999 (%)",
        "IPCA 12 meses, dez/1998 (%)", "IPCA 12 meses, dez/1999 (%)",
        "Dívida líq. GF+BC, dez/1998 (% PIB)", "Dívida líq. GF+BC, jan/1999 (% PIB)",
        "Dívida líq. GF+BC, dez/1999 (% PIB)",
        "PTAX, pico depois da flutuação (R$/US$)", "Selic, pico em 1999 (% a.a.)",
        "EMBI+ Brasil, pico set/98–dez/99 (p.b.)",
    ],
    "valor": [
        pib.loc["1998-01-01"], pib.loc["1999-01-01"],
        ipca.loc["1998-12-01"], ipca.loc["1999-12-01"],
        fis.loc["1998-12-01", "divida_pib"], fis.loc["1999-01-01", "divida_pib"],
        fis.loc["1999-12-01", "divida_pib"],
        ptax_pico[0], selic_pico[0], embi_pico[0],
    ],
    "data_ou_fonte": [
        "SGS 7326", "SGS 7326", "SGS 13522", "SGS 13522",
        "SGS 4468 / PIB 12m", "SGS 4468 / PIB 12m", "SGS 4468 / PIB 12m",
        str(ptax_pico[1]), str(selic_pico[1]), str(embi_pico[1]),
    ],
})
custos["valor"] = custos["valor"].round(2)
custos.to_csv(SAIDAS / "resposta_custos.csv", index=False)
print(custos.to_string(index=False))

# ---------- gráfico 1: virada de regime ----------
EVENTOS = {"1998-11-13": "Memorando FMI", "1998-12-02": "Stand-by", "1999-01-15": "Flutuação",
           "1999-03-04": "Selic 45%", "1999-07-01": "Metas de inflação"}
fig, axs = am.figura(2, 2, figsize=(11, 7.2), sharex=True)
paineis = [("ptax", r"Câmbio R\$/US\$ (PTAX)", am.NAVY), ("selic", "Selic efetiva (% a.a.)", am.AMBER),
           ("embi", "EMBI+ Brasil (p.b.)", am.RED), ("reservas", r"Reservas, liquidez (US\$ mi)", am.BLUE)]
for ax, (col, tit, cor) in zip(axs.flat, paineis):
    am.eixo(ax)
    ax.plot(z.index, z[col], color=cor, lw=1.5)
    for dt in EVENTOS:
        ax.axvline(pd.Timestamp(dt), color=am.MUTED, lw=0.6, ls=":")
    am.subtitulo_eixo(ax, tit)
ax = axs[0, 0]
for dt, nome in EVENTOS.items():
    ax.text(pd.Timestamp(dt), ax.get_ylim()[1], " " + nome, rotation=90, va="top",
            fontsize=6.5, color=am.INK_SOFT)
ax = axs[1, 1]
for dt, v in saltos.items():
    ax.annotate(f"+{am.br(v / 1000, 1)} bi", xy=(dt, z.loc[dt, "reservas"]),
                xytext=(8, 10), textcoords="offset points", fontsize=7, color=am.INK_SOFT,
                arrowprops=dict(arrowstyle="->", color=am.INK_SOFT, lw=0.6))
am.titulos(fig, "A resposta de política: do pacote do FMI à âncora de inflação",
           "Diário, set/1998–dez/1999; linhas pontilhadas: eventos documentados")
am.rodape(fig, "Fonte: BCB/SGS (1, 1178, 13621), Ipeadata (EMBI+); eventos: FMI (PR 98/59, "
          "memorandos de 13/11/98, 08/03/99, 02/07/99), BIS (10/12/98). Elaboração própria.")
for a in axs.flat:
    eixo_mes(a, [1, 4, 7, 10])
am.salvar(fig, SAIDAS / "resposta_virada_regime.png")

# ---------- gráfico 2: fiscal e inflação contra as metas ----------
fig, axs = am.figura(1, 2, figsize=(11, 5.4))
ax = axs[0]; am.eixo(ax)
pf = prim_fed.loc["1997-07":"2000-12"]
ax.plot(pf.index, pf, color=am.NAVY, lw=2, label="Primário GF+BC, 12 meses")
ax.hlines(2.3, pd.Timestamp("1999-01-01"), pd.Timestamp("1999-12-01"), color=am.RED,
          lw=1.5, ls="--", label="Meta federal 1999 (≥ 2,3%, memorando de mar/99)")
ax.axhline(0, color=am.MUTED, lw=0.8)
ax.axvline(pd.Timestamp("1999-01-15"), color=am.MUTED, lw=0.6, ls=":")
fator = 25.4 / fis.loc["1998-12-01", "divida_pib"]   # PIB da época / PIB revisado (dez/98)
ax.text(pd.Timestamp("1997-08-01"), pf.max() * 0.62,
        f"Com o PIB da época (÷{am.br(1 / fator, 2)}), dez/99 = "
        f"{am.br(pf.loc['1999-12-01'] * fator, 2)}%.\nConsolidado 1999: meta R\\$ 30,2 bi;"
        " realizado R\\$ 31,1 bi (FMI).", fontsize=7, color=am.INK_SOFT)
am.subtitulo_eixo(ax, "Superávit primário federal (% PIB revisado)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax = axs[1]; am.eixo(ax)
ip = ipca.loc["1997-07":"2000-12"]
ax.plot(ip.index, ip, color=am.NAVY, lw=2, label="IPCA, 12 meses")
for ano, meta in [(1999, 8), (2000, 6)]:
    x0, x1 = pd.Timestamp(f"{ano}-01-01"), pd.Timestamp(f"{ano}-12-01")
    ax.fill_between([x0, x1], meta - 2, meta + 2, color=am.BLUE_WASH, alpha=0.7)
    ax.hlines(meta, x0, x1, color=am.BLUE, lw=1.2)
    ax.text(x0, meta + 2.2, f"meta {ano}: {meta}% ± 2", fontsize=7, color=am.BLUE)
ax.axvline(pd.Timestamp("1999-01-15"), color=am.MUTED, lw=0.6, ls=":")
am.subtitulo_eixo(ax, "Inflação contra a meta (%)")
am.titulos(fig, "O ajuste prometido ao FMI foi entregue, e a inflação ficou na meta",
           "Primário federal e IPCA, 1997–2000")
am.rodape(fig, "Fonte: BCB/SGS (5783, 13522); metas: FMI, memorandos de 08/03/1999 (par. 11) e "
          "02/07/1999 (par. 17). Elaboração própria.")
for a in axs.flat:
    eixo_mes(a, [1, 7])
fig.subplots_adjust(top=0.80)
am.salvar(fig, SAIDAS / "resposta_fiscal_inflacao.png")
