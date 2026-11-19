"""Controle sintético (Abadie, Diamond e Hainmueller, 2010): o custo do colapso.

Pergunta (blocos 5–6): quanto o colapso da banda custou em atividade, e quanto
a desvalorização passou para a inflação, comparado a um "Brasil sintético" que
não sofreu a crise?

Resultados (mensais, OECD/MEI via FRED):
  - produção manufatureira (índice, sem ajuste sazonal → STL em log,
    normalizado para média de 1997 = 100): PRMNTO01IXOBM;
  - inflação ao consumidor em 12 meses (%): CPALTT01…M659N.
Tratamento: jan/1999 (flutuação). Pré-tratamento: jan/1995–dez/1998.

Pool de doadores (principal): emergentes e economias pequenas e abertas sem
crise cambial própria em 1997–2000. Excluídos: Coreia e Indonésia (Ásia, 1997),
Rússia (1998), Chile e Colômbia (abandono de bandas/recessão em 1999),
África do Sul (crise do rand, 1998). Pool amplo (robustez): + avançadas.

Pesos: w ≥ 0, Σw = 1, minimizando o erro quadrático médio do resultado em
todos os meses do pré-tratamento.
Inferência: placebos no espaço (cada doador tratado como se fosse o Brasil):
razão RMSPE pós/pré e p-valor de permutação = posição do Brasil/(N+1);
placebo no tempo (tratamento fictício em jan/1997).

Cuidados: o Brasil já desacelerava em 1998 (juros da defesa): o efeito medido a
partir de jan/1999 é do colapso e não da defesa; tratamento em set/1998 vai
como robustez. Um único tratado: a inferência é por permutação, não assintótica.
"""
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from statsmodels.tsa.seasonal import STL

RAIZ = Path(__file__).resolve().parents[1]
DADOS, SAIDAS = RAIZ / "dados", RAIZ / "saidas"
sys.path.insert(0, str(RAIZ / "src"))
import estilo_am as am  # noqa: E402

POOL = ["MEX", "POL", "HUN", "CZE", "SVK", "SVN", "ISR", "IND", "TUR",
        "PRT", "GRC", "ESP"]
AMPLO = POOL + ["CAN", "NOR", "JPN", "USA", "DEU", "FRA", "ITA", "IRL", "FIN",
                "SWE", "DNK", "AUT", "BEL", "NLD"]
NOMES = {"BRA": "Brasil", "MEX": "México", "POL": "Polônia", "HUN": "Hungria",
         "CZE": "Rep. Tcheca", "SVK": "Eslováquia", "SVN": "Eslovênia", "ISR": "Israel",
         "IND": "Índia", "TUR": "Turquia", "PRT": "Portugal", "GRC": "Grécia",
         "ESP": "Espanha", "CAN": "Canadá", "NOR": "Noruega", "JPN": "Japão",
         "USA": "EUA", "DEU": "Alemanha", "FRA": "França", "ITA": "Itália",
         "IRL": "Irlanda", "FIN": "Finlândia", "SWE": "Suécia", "DNK": "Dinamarca",
         "AUT": "Áustria", "BEL": "Bélgica", "NLD": "Holanda"}


def fred(serie: str) -> pd.Series:
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                     params={"id": serie, "cosd": "1994-01-01", "coed": "2001-12-31"},
                     timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
    return pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                     index=df["observation_date"])


ISO2 = {"BRA": "BR", "MEX": "MX", "POL": "PL", "HUN": "HU", "CZE": "CZ", "SVK": "SK",
        "SVN": "SI", "ISR": "IL", "IND": "IN", "TUR": "TR", "PRT": "PT", "GRC": "GR",
        "ESP": "ES", "CAN": "CA", "NOR": "NO", "JPN": "JP", "USA": "US", "DEU": "DE",
        "FRA": "FR", "ITA": "IT", "IRL": "IE", "FIN": "FI", "SWE": "SE", "DNK": "DK",
        "AUT": "AT", "BEL": "BE", "NLD": "NL"}


def coletar(sufixo: str, arquivo: str) -> pd.DataFrame:
    caminho = DADOS / arquivo
    if caminho.exists():
        return pd.read_csv(caminho, index_col=0, parse_dates=True)
    series = {}
    for c in ["BRA"] + AMPLO:
        try:
            series[c] = fred(sufixo.format(c=c, c2=ISO2[c]))
        except requests.HTTPError:
            print(f"sem série {sufixo.format(c=c, c2=ISO2[c])}")
    df = pd.DataFrame(series)
    df.to_csv(caminho, index_label="mes")
    return df


prod_bruta = coletar("{c}PRMNTO01IXOBM", "sc_producao_manufatureira.csv")
infl = coletar("CPALTT01{c2}M659N", "sc_inflacao_12m.csv")


def dessazonalizar(s: pd.Series) -> pd.Series:
    s = np.log(s.dropna())
    sa = s - STL(s, period=12, robust=True).fit().seasonal
    return 100 * np.exp(sa - sa.loc["1997"].mean())


prod = pd.DataFrame({c: dessazonalizar(prod_bruta[c]) for c in prod_bruta.columns})
prod.to_csv(SAIDAS / "sc_producao_dessaz.csv", index_label="mes")


def sintetico(y: pd.DataFrame, tratado: str, doadores: list, ini: str, t0: str):
    pre = y.loc[ini:t0].index[:-1] if False else y.loc[ini:pd.Timestamp(t0) - pd.offsets.MonthBegin(1)].index
    Y1 = y.loc[pre, tratado].values
    Y0 = y.loc[pre, doadores].values
    n = len(doadores)
    obj = lambda w: np.mean((Y1 - Y0 @ w) ** 2)
    res = minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
                   options={"maxiter": 500, "ftol": 1e-12})
    w = pd.Series(res.x, index=doadores).clip(lower=0)
    w /= w.sum()
    synth = y[doadores] @ w
    return w, synth


def rmspe(gap, ini, fim):
    g = gap.loc[ini:fim]
    return float(np.sqrt((g ** 2).mean()))


def estudo(y, nome, t0="1999-01-01", ini="1995-01-01", fim="2000-12-01",
           doadores=POOL, placebos=True):
    y = y.loc[ini:fim, ["BRA"] + doadores].dropna(axis=1)
    doadores = [d for d in doadores if d in y.columns]
    w, s = sintetico(y, "BRA", doadores, ini, t0)
    gap = y["BRA"] - s
    pre_fim = pd.Timestamp(t0) - pd.offsets.MonthBegin(1)
    out = {"pesos": w, "sint": s, "gap": gap,
           "rmspe_pre": rmspe(gap, ini, pre_fim), "rmspe_pos": rmspe(gap, t0, fim)}
    if placebos:
        gaps, razoes = {}, {"BRA": out["rmspe_pos"] / out["rmspe_pre"]}
        for d in doadores:
            outros = [x for x in doadores if x != d]
            _, sd = sintetico(y, d, outros, ini, t0)
            g = y[d] - sd
            gaps[d] = g
            razoes[d] = rmspe(g, t0, fim) / rmspe(g, ini, pre_fim)
        out["gaps_placebo"] = pd.DataFrame(gaps)
        out["razoes"] = pd.Series(razoes).sort_values(ascending=False)
        out["p_valor"] = (out["razoes"].index.get_loc("BRA") + 1) / len(out["razoes"])
    return out


resultados = {}
linhas = []


def registrar(rot, out, y, t0="1999-01-01", fim="2000-12-01"):
    g = out["gap"]
    linhas.append({
        "especificação": rot,
        "rmspe_pré": round(out["rmspe_pre"], 2),
        "gap médio 1999": round(g.loc["1999"].mean(), 2),
        "gap médio 2000": round(g.loc["2000"].mean(), 2) if fim >= "2000-12-01" else np.nan,
        "gap médio jan/99–dez/00": round(g.loc[t0:fim].mean(), 2),
        "p-valor (permutação)": round(out.get("p_valor", np.nan), 3),
        "doadores com peso > 5%": ", ".join(
            f"{NOMES[k]} {v:.0%}" for k, v in out["pesos"].sort_values(ascending=False).items()
            if v > 0.05),
    })
    resultados[rot] = out


registrar("Produção: pool principal, tratamento jan/99", estudo(prod, "prod"), prod)
registrar("Produção: pool amplo", estudo(prod, "prod", doadores=AMPLO), prod)
registrar("Produção: tratamento set/98 (defesa)",
          estudo(prod, "prod", t0="1998-09-01"), prod, t0="1998-09-01")
placebo_tempo = estudo(prod, "prod", t0="1997-01-01", fim="1998-08-01", placebos=False)
linhas.append({"especificação": "Produção: placebo no tempo (jan/97, até ago/98)",
               "rmspe_pré": round(placebo_tempo["rmspe_pre"], 2),
               "gap médio jan/99–dez/00": round(placebo_tempo["gap"].loc["1997-01":"1998-08"].mean(), 2)})
resultados["placebo_tempo"] = placebo_tempo
registrar("Produção: pré-tratamento desde jan/96 (pós-Tequila)",
          estudo(prod, "prod", ini="1996-01-01"), prod)
# Inflação: o pré-tratamento começa em 1997, quando a inflação brasileira já
# estava estabilizada (1995–96 ainda carregam a desinflação do Real, fora do
# alcance de qualquer combinação de doadores).
registrar("Inflação 12m: pool principal, pré desde jan/97",
          estudo(infl, "infl", ini="1997-01-01"), infl)

tab = pd.DataFrame(linhas).set_index("especificação")
tab.to_csv(SAIDAS / "sc_resultados.csv")
pd.set_option("display.width", 250, "display.max_colwidth", 70)
print(tab.to_string())
base = resultados["Produção: pool principal, tratamento jan/99"]
print("\nPesos (produção, pool principal):")
print(base["pesos"].sort_values(ascending=False).round(3).to_string())
print("\nRazões RMSPE pós/pré (produção):")
print(base["razoes"].round(2).to_string())
inf_base = resultados["Inflação 12m: pool principal, pré desde jan/97"]
print("\nPesos (inflação):", inf_base["pesos"][inf_base["pesos"] > 0.01].round(3).to_dict())
print("Razões (inflação):", inf_base["razoes"].round(2).head(5).to_dict())

# ---------- gráficos ----------
T0 = pd.Timestamp("1999-01-01")
fig, axs = am.figura(1, 2, figsize=(11, 4.8))
ax = axs[0]; am.eixo(ax)
y = prod.loc["1995":"2000", "BRA"]
ax.plot(y.index, y, color=am.NAVY, lw=2, label="Brasil")
ax.plot(base["sint"].loc["1995":"2000"].index, base["sint"].loc["1995":"2000"],
        color=am.BLUE, lw=1.8, ls="--", label="Brasil sintético")
ax.axvline(T0, color=am.RED, lw=0.8, ls=":")
ax.text(T0, ax.get_ylim()[1], " flutuação", color=am.RED, fontsize=7, va="top")
am.subtitulo_eixo(ax, "Produção manufatureira (dessaz., 1997 = 100)")
ax.legend(frameon=False, fontsize=8, loc="lower left")

ax = axs[1]; am.eixo(ax)
pre_br = base["rmspe_pre"]
for d, g in base["gaps_placebo"].items():
    pre_d = rmspe(g, "1995-01-01", "1998-12-01")
    if pre_d <= 5 * pre_br:
        ax.plot(g.index, g, color=am.MUTED, lw=0.7, alpha=0.6)
ax.plot(base["gap"].index, base["gap"], color=am.NAVY, lw=2.2, label="Brasil")
ax.plot([], [], color=am.MUTED, lw=0.7, label="Placebos (doadores como tratados)")
ax.axhline(0, color=am.MUTED, lw=0.8)
ax.axvline(T0, color=am.RED, lw=0.8, ls=":")
am.subtitulo_eixo(ax, f"Diferença real − sintético (p-valor = {am.br(base['p_valor'], 2)})")
ax.legend(frameon=False, fontsize=8, loc="lower left")
am.titulos(fig, "Controle sintético: o custo do colapso na indústria",
           "Brasil contra uma combinação de emergentes sem crise cambial própria, 1995–2000")
am.rodape(fig, "Fonte: OECD/MEI via FRED (PRMNTO01IXOBM). Pesos que minimizam o erro de jan/95 a "
          "dez/98; placebos com RMSPE pré até 5× o do Brasil. Elaboração própria.")
fig.subplots_adjust(top=0.80)
am.salvar(fig, SAIDAS / "sc_producao.png")

fig, ax = am.figura(1, 1, figsize=(9.5, 4.6))
am.eixo(ax)
yi = infl.loc["1997":"2000", "BRA"]
ax.plot(yi.index, yi, color=am.NAVY, lw=2, label="Brasil")
ax.plot(inf_base["sint"].loc["1997":"2000"].index, inf_base["sint"].loc["1997":"2000"],
        color=am.BLUE, lw=1.8, ls="--", label="Brasil sintético")
ax.axvline(T0, color=am.RED, lw=0.8, ls=":")
am.subtitulo_eixo(ax, f"Inflação ao consumidor, 12 meses (%); p-valor = {am.br(inf_base['p_valor'], 2)}")
ax.legend(frameon=False, fontsize=8)
am.titulos(fig, "Controle sintético: o repasse da desvalorização à inflação",
           "Brasil contra emergentes sem crise cambial própria")
am.rodape(fig, "Fonte: OECD/MEI via FRED (CPALTT01…M659N). Elaboração própria.")
fig.subplots_adjust(top=0.80)
am.salvar(fig, SAIDAS / "sc_inflacao.png")
