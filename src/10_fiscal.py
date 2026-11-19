"""Bloco 1–2 do roteiro: a política fiscal antes do colapso da banda.

Pergunta: a política fiscal era consistente com a âncora cambial? E quanto a
defesa da banda com juros custava ao fiscal?

Nível: Governo Federal + Banco Central (o consolidado só existe no SGS a partir
de 2001-02). Sinal das NFSP: positivo = déficit; aqui usamos s = superávit
primário = −NFSP primário.

1. Trajetória (12 meses, % PIB): primário, juros nominais, nominal; dívida
   líquida/PIB. A dívida/PIB é calculada com o PIB de 12 meses valorizado pelo
   IGP-DI do mês (método do BC).
2. Decomposição da variação anual da dívida/PIB (dez–dez):
      Δd = primário + juros − d₋₁·γ/(1+γ) + ajustes
   γ = crescimento nominal do PIB; "ajustes" = resíduo (câmbio sobre a dívida
   indexada e externa, reconhecimento de passivos, privatização).
3. Função de reação fiscal (Bohn, 1998): s_t = α + ρ·d_{t−12} + ε_t.
   ρ > 0 é condição suficiente de sustentabilidade. Testamos se ρ muda em 1999
   (acordo com o FMI e metas de primário). Erros HAC (Newey-West, 12 defasagens),
   porque os fluxos de 12 meses se sobrepõem.
4. Custo da defesa com juros: taxa implícita da dívida líquida federal
   (juros do mês / dívida do mês anterior) contra a Selic. A inclinação mede
   quanto da Selic passa ao custo da dívida; multiplicada pela dívida/PIB dá o
   custo fiscal de cada ponto de Selic.
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

f = pd.read_csv(DADOS / "fiscal_mensal.csv", index_col=0, parse_dates=True)

# ---------- PIB de 12 meses valorizado pelo IGP-DI ----------
igp = (1 + f["igpdi_var"] / 100).cumprod()
pib_v = pd.Series(np.nan, index=f.index)
for t in range(11, len(f)):
    jan = slice(t - 11, t + 1)
    pib_v.iloc[t] = (f["pib_mensal"].iloc[jan] * igp.iloc[t] / igp.iloc[jan]).sum()
f["pib12_valorizado"] = pib_v
f["divida_pib"] = f["dlsp_gf"] / f["pib12_valorizado"] * 100
f["superavit_12m"] = -f["primario_gf_12m_pib"]
f["juros_12m"] = f["juros_gf_12m_pib"]
f["nominal_12m"] = f["nominal_gf_12m_pib"]

# ---------- 2. decomposição anual ----------
dez = f[f.index.month == 12].copy()
dez["pib12_nom"] = f["pib_mensal"].rolling(12).sum().reindex(dez.index)
linhas = []
for i in range(1, len(dez)):
    t, t0 = dez.index[i], dez.index[i - 1]
    if t.year < 1993 or t.year > 2002:
        continue
    d, d0 = dez.loc[t, "divida_pib"], dez.loc[t0, "divida_pib"]
    gam = dez.loc[t, "pib12_valorizado"] / dez.loc[t0, "pib12_valorizado"] - 1
    prim = dez.loc[t, "primario_gf_12m_pib"]       # + = déficit
    jur = dez.loc[t, "juros_12m"]
    cresc = -d0 * gam / (1 + gam)
    linhas.append({"ano": t.year, "divida_pib_dez": round(d, 1),
                   "variacao": round(d - d0, 1), "primario": round(prim, 1),
                   "juros": round(jur, 1), "crescimento_pib": round(cresc, 1),
                   "ajustes_residuo": round(d - d0 - prim - jur - cresc, 1),
                   # primário que estabilizaria a dívida/PIB sem ajustes:
                   # s* = juros − d₋₁·γ/(1+γ)
                   "primario_necessario": round(jur + cresc, 1),
                   "primario_efetivo": round(-prim, 1),
                   "hiato_fiscal": round(jur + cresc + prim, 1)})
decomp = pd.DataFrame(linhas).set_index("ano")
decomp.to_csv(SAIDAS / "fiscal_decomposicao_divida.csv")
pd.set_option("display.width", 200)
print("Decomposição da variação da dívida líquida federal (p.p. do PIB, dez-dez):")
print(decomp.to_string())

# ---------- 3. função de reação de Bohn ----------
b = pd.DataFrame({"s": f["superavit_12m"], "d_lag": f["divida_pib"].shift(12)}).dropna()
b = b.loc["1995-01":"2006-12"]
b["pos99"] = (b.index >= "1999-01-01").astype(int)
b["d_x_pos99"] = b["d_lag"] * b["pos99"]


def hac(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})


r_pre = hac(b.loc[:"1998-12", "s"], b.loc[:"1998-12", ["d_lag"]])
r_pos = hac(b.loc["1999-01":, "s"], b.loc["1999-01":, ["d_lag"]])
r_int = hac(b["s"], b[["d_lag", "pos99", "d_x_pos99"]])
bohn = pd.DataFrame({
    "1995–1998": [r_pre.params["d_lag"], r_pre.bse["d_lag"], r_pre.pvalues["d_lag"],
                  r_pre.params["const"], int(r_pre.nobs), r_pre.rsquared],
    "1999–2006": [r_pos.params["d_lag"], r_pos.bse["d_lag"], r_pos.pvalues["d_lag"],
                  r_pos.params["const"], int(r_pos.nobs), r_pos.rsquared],
}, index=["rho", "ep_HAC", "p", "constante", "n", "R2"]).round(3)
bohn.loc["s médio (% PIB)"] = [round(b.loc[:"1998-12", "s"].mean(), 2),
                               round(b.loc["1999-01":, "s"].mean(), 2)]
bohn.to_csv(SAIDAS / "fiscal_bohn.csv")
print("\nFunção de reação fiscal (s = superávit primário GF+BC, 12m, % PIB; "
      "d = dívida líquida GF+BC/PIB defasada 12 meses):")
print(bohn)
print(f"Quebra em 1999 (interação d×pós-99): {r_int.params['d_x_pos99']:.3f} "
      f"(p={r_int.pvalues['d_x_pos99']:.3f}); salto de nível pós-99: "
      f"{r_int.params['pos99']:.2f} (p={r_int.pvalues['pos99']:.3f})")

# ---------- 4. custo fiscal da defesa com juros ----------
d_dia = pd.read_csv(DADOS / "painel_diario.csv", index_col=0, parse_dates=True)
selic_m = d_dia["selic"].resample("MS").mean()
impl = (1 + f["juros_gf"] / f["dlsp_gf"].shift(1)) ** 12 - 1
c = pd.DataFrame({"taxa_implicita": impl * 100, "selic": selic_m}).dropna()
c = c.loc["1995-01":"1998-12"]
custo = {}
for nome, jan in [("1995–1997", slice("1995-01", "1997-12")),
                  ("1998", slice("1998-01", "1998-12")),
                  ("1995–1998", slice("1995-01", "1998-12"))]:
    r = hac(c.loc[jan, "taxa_implicita"], c.loc[jan, ["selic"]])
    custo[nome] = {"inclinacao": r.params["selic"], "ep_HAC": r.bse["selic"],
                   "p": r.pvalues["selic"], "n": int(r.nobs), "R2": r.rsquared}
custo = pd.DataFrame(custo).round(3)
d98 = f.loc["1998-08-01", "divida_pib"]
alta = selic_m.loc["1998-10-01"] - selic_m.loc["1998-08-01"]
for col in custo.columns:
    beta = custo.loc["inclinacao", col]
    custo.loc["custo de +10 p.p. de Selic (% PIB/ano)", col] = round(beta * 10 / 100 * d98, 2)
    custo.loc[f"custo da alta ago→out/98 (+{alta:.1f} p.p.) (% PIB/ano)", col] = round(
        beta * alta / 100 * d98, 2)
custo.to_csv(SAIDAS / "fiscal_custo_juros.csv")
print(f"\nTaxa implícita da dívida líquida federal × Selic (dívida/PIB em ago/98 = {d98:.1f}%):")
print(custo)
c.to_csv(SAIDAS / "fiscal_taxa_implicita.csv", index_label="mes")

# controle: valores de dez/1998 contra o Relatório Anual do BC (1999, Quadro 4.4)
print("\nControle dez/1998: dívida GF+BC R$ mi =", round(f.loc["1998-12-01", "dlsp_gf"]),
      "(Relatório: 231 268); dívida/PIB =", round(f.loc["1998-12-01", "divida_pib"], 1),
      "(Relatório: 25,4, com PIB antigo)")
f.to_csv(SAIDAS / "fiscal_painel.csv", index_label="mes")

# ---------- gráficos ----------
fig, axs = am.figura(2, 2, figsize=(13, 6.6))
ax = axs[0, 0]; am.eixo(ax)
p = f.loc["1995-07":"2001-12"]
ax.plot(p.index, p["juros_12m"], color=am.AMBER, lw=1.8, label="Juros nominais")
ax.plot(p.index, p["nominal_12m"], color=am.NAVY, lw=1.8, label="Déficit nominal")
ax.plot(p.index, -p["superavit_12m"], color=am.BLUE, lw=1.8, label="Déficit primário")
ax.axhline(0, color=am.MUTED, lw=0.8)
ax.axvline(pd.Timestamp("1999-01-15"), color=am.RED, lw=0.7, ls="--")
am.subtitulo_eixo(ax, "NFSP do Gov. Federal + BC, 12 meses (% PIB)")
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axs[0, 1]; am.eixo(ax)
ax.plot(p.index, p["divida_pib"], color=am.NAVY, lw=2)
ax.axvline(pd.Timestamp("1999-01-15"), color=am.RED, lw=0.7, ls="--")
ax.text(pd.Timestamp("1999-01-15"), p["divida_pib"].min(), " flutuação", color=am.RED, fontsize=7)
am.subtitulo_eixo(ax, "Dívida líquida do Gov. Federal + BC (% PIB)")

ax = axs[1, 0]; am.eixo(ax)
dd = decomp.loc[1995:1999]
cores = {"primario": am.BLUE, "juros": am.AMBER, "crescimento_pib": am.GREEN,
         "ajustes_residuo": am.MUTED}
nomes = {"primario": "Primário", "juros": "Juros", "crescimento_pib": "Crescimento do PIB",
         "ajustes_residuo": "Câmbio e outros ajustes"}
pos = np.zeros(len(dd)); neg = np.zeros(len(dd))
for col, cor in cores.items():
    v = dd[col].values
    base = np.where(v >= 0, pos, neg)
    ax.bar(dd.index, v, bottom=base, color=cor, label=nomes[col], width=0.6)
    pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
ax.plot(dd.index, dd["variacao"], "o", color=am.RED, ms=5, label="Variação total")
ax.axhline(0, color=am.MUTED, lw=0.8)
am.subtitulo_eixo(ax, "Variação da dívida/PIB por fator (p.p., dez–dez)")
ax.legend(frameon=False, fontsize=7, loc="upper left")

ax = axs[1, 1]; am.eixo(ax)
for nome, jan, cor in [("1995–1997", slice("1995-01", "1997-12"), am.BLUE),
                       ("1998", slice("1998-01", "1998-12"), am.RED)]:
    cc = c.loc[jan]
    r = hac(cc["taxa_implicita"], cc[["selic"]])
    ax.scatter(cc["selic"], cc["taxa_implicita"], color=cor, s=18,
               label=f"{nome}: inclinação {am.br(r.params['selic'], 2)}")
    xs = np.linspace(cc["selic"].min(), cc["selic"].max(), 10)
    ax.plot(xs, r.params["const"] + r.params["selic"] * xs, color=cor, lw=1.2)
am.subtitulo_eixo(ax, "Taxa implícita da dívida federal × Selic (% a.a.)")
ax.set_xlabel("Selic média do mês (% a.a.)", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left")
am.titulos(fig, "Política fiscal: déficit nominal alto, dívida em alta e juros no comando",
           "Governo Federal + Banco Central, 1994–2001")
am.rodape(fig, "Fonte: BCB/SGS (NFSP 4639, 4573, 4606, 5783, 5717, 5750; DLSP 4468; PIB 4380; "
          "IGP-DI 190; Selic 1178). Elaboração própria.")
for a in (axs[0, 0], axs[0, 1]):
    am.eixo_datas(a)
fig.subplots_adjust(top=0.86, hspace=0.35, wspace=0.15)
am.salvar(fig, SAIDAS / "fiscal_painel.png")
