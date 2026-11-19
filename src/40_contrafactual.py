"""Bloco 6 do roteiro: contrafactual.

Pergunta: que políticas teriam reduzido a probabilidade ou a severidade da crise?

Método: simulação contábil da dívida líquida do Governo Federal + BC, mensal,
jan/1996–dez/1999, calibrada para reproduzir EXATAMENTE a trajetória observada:

    D_t = D_{t−1}·(1 + r_t) + P_t + A_t

D = dívida líquida (SGS 4468); r = taxa implícita do mês = juros_t / D_{t−1}
(SGS 4606); P = NFSP primária do mês (SGS 4639, + = déficit); A = resíduo
(câmbio sobre a dívida indexada, reconhecimento de passivos, privatização).
Cada cenário troca UM elemento e mantém o resto igual ao observado.

Cenários:
  1. Primário de 1999 antecipado: superávit de 1999 (% do PIB) desde jan/1996.
  2. Dívida menos indexada à Selic: em 1998 a taxa implícita reage à Selic com
     a sensibilidade de 1995–97 (estimada em 10_fiscal.py), e não com a de 1998.
     Só o desvio da Selic em relação à média calma de jan–set/1997 é afetado.
  3. Sem as defesas com juros (Ásia e Rússia, nov/1997–dez/1998): a Selic fica
     na média calma de jan–set/1997; a desvalorização de 1999 é a mesma.
  4. Cenários 1 + 2.

Limites: equilíbrio parcial. Selic, câmbio, crescimento e o resíduo A são
mantidos como observados; um primário maior provavelmente teria reduzido
também o prêmio de risco e os juros (o contrafactual subestima o ganho), e a
flutuação antecipada teria antecipado o salto cambial sobre a dívida indexada.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

RAIZ = Path(__file__).resolve().parents[1]
DADOS, SAIDAS = RAIZ / "dados", RAIZ / "saidas"
sys.path.insert(0, str(RAIZ / "src"))
import estilo_am as am  # noqa: E402

f = pd.read_csv(SAIDAS / "fiscal_painel.csv", index_col=0, parse_dates=True)
d_dia = pd.read_csv(DADOS / "painel_diario.csv", index_col=0, parse_dates=True)
selic = d_dia["selic"].resample("MS").mean()

INI, FIM = "1996-01-01", "1999-12-01"
m = pd.DataFrame({
    "D": f["dlsp_gf"], "J": f["juros_gf"], "P": f["primario_gf"],
    "pib": f["pib_mensal"], "pibv": f["pib12_valorizado"], "selic": selic,
})
m["D0"] = m["D"].shift(1)
m["r"] = m["J"] / m["D0"]
m["A"] = m["D"] - m["D0"] - m["J"] - m["P"]
m = m.loc[INI:FIM].copy()

# ---- sensibilidades da taxa implícita à Selic (mesma regressão de 10_fiscal) ----
impl = ((1 + f["juros_gf"] / f["dlsp_gf"].shift(1)) ** 12 - 1) * 100
c = pd.DataFrame({"impl": impl, "selic": selic}).dropna()


def inclinacao(jan):
    cc = c.loc[jan]
    return sm.OLS(cc["impl"], sm.add_constant(cc["selic"])).fit().params["selic"]


b95, b98 = inclinacao(slice("1995-01", "1997-12")), inclinacao(slice("1998-01", "1998-12"))
selic_ref = selic.loc["1997-01":"1997-09"].mean()
s99 = -f.loc["1999-12-01", "primario_gf_12m_pib"]      # superávit de 1999, % PIB
print(f"Sensibilidade à Selic: 1995–97 = {b95:.3f}; 1998 = {b98:.3f}; "
      f"Selic de referência (jan–set/97, antes da Ásia) = {selic_ref:.1f}%; primário 1999 = {s99:.2f}% PIB")


def anual_para_mensal(taxa_aa_pct):
    return (1 + taxa_aa_pct / 100) ** (1 / 12) - 1


def simular(r=None, P=None):
    r = m["r"] if r is None else r
    P = m["P"] if P is None else P
    D = []
    prev = m["D0"].iloc[0]
    for t in range(len(m)):
        prev = prev * (1 + r.iloc[t]) + P.iloc[t] + m["A"].iloc[t]
        D.append(prev)
    return pd.Series(D, index=m.index)


def r_ajustada(delta_aa: pd.Series) -> pd.Series:
    """Subtrai `delta_aa` (p.p. ao ano) da taxa implícita anualizada do mês."""
    r_aa = ((1 + m["r"]) ** 12 - 1) * 100
    return anual_para_mensal(r_aa - delta_aa.reindex(m.index).fillna(0))


desvio = (m["selic"] - selic_ref)
em98 = (m.index >= "1998-01-01") & (m.index <= "1998-12-01")
set_dez98 = (m.index >= "1997-11-01") & (m.index <= "1998-12-01")  # defesas Ásia + Rússia

cen = {}
cen["Observado"] = simular()
cen["1. Primário de 1999 desde 1996"] = simular(P=-s99 / 100 * m["pib"])
cen["2. Dívida menos indexada à Selic (1998)"] = simular(
    r=r_ajustada(((b98 - b95) * desvio).where(em98, 0)))
cen["3. Sem as defesas com juros (nov/97–dez/98)"] = simular(
    r=r_ajustada((b98 * desvio).where(set_dez98, 0)))
cen["4. Cenários 1 + 2"] = simular(
    P=-s99 / 100 * m["pib"], r=r_ajustada(((b98 - b95) * desvio).where(em98, 0)))

assert np.allclose(cen["Observado"], m["D"]), "a simulação base não reproduz a dívida"

# base para o simulador do app (app/dados/)
import json  # noqa: E402
APP = RAIZ / "app" / "dados"
APP.mkdir(parents=True, exist_ok=True)
m.to_csv(APP / "contrafactual_base.csv", index_label="mes")
json.dump({"b95": b95, "b98": b98, "selic_ref": selic_ref, "s99": s99},
          open(APP / "parametros.json", "w"), indent=2)
dp = pd.DataFrame({k: v / m["pibv"] * 100 for k, v in cen.items()})
dp.to_csv(SAIDAS / "contrafactual_divida_pib.csv", index_label="mes")

tab = pd.DataFrame({
    "dez/1997": dp.loc["1997-12-01"], "dez/1998": dp.loc["1998-12-01"],
    "dez/1999": dp.loc["1999-12-01"],
}).round(1)
tab["dif. dez/1998 (p.p.)"] = (tab["dez/1998"] - tab.loc["Observado", "dez/1998"]).round(1)
tab["dif. dez/1999 (p.p.)"] = (tab["dez/1999"] - tab.loc["Observado", "dez/1999"]).round(1)
# juros pagos a menos em 1998 (% do PIB de 1998)
pib98 = m.loc["1998-01":"1998-12", "pib"].sum()
for nome, serie in cen.items():
    D0 = serie.shift(1).fillna(m["D0"].iloc[0])
    if nome.startswith("2"):
        r = r_ajustada(((b98 - b95) * desvio).where(em98, 0))
    elif nome.startswith("3"):
        r = r_ajustada((b98 * desvio).where(set_dez98, 0))
    elif nome.startswith("4"):
        r = r_ajustada(((b98 - b95) * desvio).where(em98, 0))
    else:
        r = m["r"]
    tab.loc[nome, "juros 1998 (% PIB)"] = round(
        (D0 * r).loc["1998-01":"1998-12"].sum() / pib98 * 100, 2)
    tab.loc[nome, "juros nov/97–dez/98 (R$ bi)"] = round(
        (D0 * r).loc["1997-11":"1998-12"].sum() / 1000, 1)
tab.to_csv(SAIDAS / "contrafactual_resumo.csv")
pd.set_option("display.width", 220)
print(tab.to_string())

# ---- gráfico ----
fig, ax = am.figura(1, 1, figsize=(9.5, 5.4))
am.eixo(ax)
cores = {"Observado": am.NAVY, "1. Primário de 1999 desde 1996": am.GREEN,
         "2. Dívida menos indexada à Selic (1998)": am.AMBER,
         "3. Sem as defesas com juros (nov/97–dez/98)": am.BLUE,
         "4. Cenários 1 + 2": am.RED}
for nome, cor in cores.items():
    ax.plot(dp.index, dp[nome], color=cor, lw=2.4 if nome == "Observado" else 1.5,
            ls="-" if nome == "Observado" else "--",
            label=f"{nome}: dez/98 {am.br(dp.loc['1998-12-01', nome])}%")
for dt, nome in {"1997-10-23": "Ásia", "1998-08-17": "Rússia", "1999-01-15": "Flutuação"}.items():
    ax.axvline(pd.Timestamp(dt), color=am.MUTED, lw=0.7, ls=":")
    ax.text(pd.Timestamp(dt), ax.get_ylim()[1], " " + nome, fontsize=7, color=am.INK_SOFT, va="top")
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.92))
am.titulos(fig, "Contrafactual: dívida líquida do Gov. Federal + BC (% do PIB)",
           "Simulação contábil que reproduz a dívida observada e troca um elemento por vez")
am.rodape(fig, "Fonte: BCB/SGS (4468, 4606, 4639, 4380, 190, 1178). Equilíbrio parcial: Selic, "
          "câmbio e crescimento como observados. Elaboração própria.")
am.salvar(fig, SAIDAS / "contrafactual_divida.png")
