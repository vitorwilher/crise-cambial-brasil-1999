"""Coleta das séries diárias da crise cambial brasileira de 1999.

Grava em dados/ os brutos de cada fonte e um painel diário `painel_diario.csv`.
Fontes: SGS/BCB (câmbio, Selic, reservas diárias 13621 e mensais 3546), FRED (Fed funds), Ipeadata (EMBI+).
"""
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
DADOS.mkdir(exist_ok=True)

INICIO, FIM = "01/07/1994", "31/12/2000"

SGS = {"ptax": 1, "selic": 1178, "reservas": 13621}


def sgs(cod: int, ini: str = INICIO, fim: str = FIM) -> pd.Series:
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados"
    r = requests.get(url, params={"formato": "json", "dataInicial": ini,
                                  "dataFinal": fim}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    s = pd.Series(pd.to_numeric(df["valor"]).values,
                  index=pd.to_datetime(df["data"], dayfirst=True))
    return s


def fred(cod: str) -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = requests.get(url, params={"id": cod, "cosd": "1994-07-01",
                                  "coed": "2000-12-31"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
    return pd.Series(pd.to_numeric(df[cod], errors="coerce").values,
                     index=df["observation_date"])


def ipea(cod: str) -> pd.Series:
    url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{cod}')"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["value"])
    idx = pd.to_datetime(df["VALDATA"].str[:10])
    s = pd.Series(pd.to_numeric(df["VALVALOR"]).values, index=idx).sort_index()
    return s.loc["1994-07-01":"2000-12-31"]


def main() -> None:
    series = {nome: sgs(cod) for nome, cod in SGS.items()}
    series["fedfunds"] = fred("DFF")
    series["embi"] = ipea("JPM366_EMBI366")

    # Reservas mensais (liquidez, US$ mi): a diária (13621) só começa em set/1998
    mensal = {"reservas_mensal": sgs(3546),
              # Prazo médio dos títulos emitidos em oferta pública (mensal; só
              # meses com emissão). LTN desde ago/1996; LFT desde jul/1998.
              "prazo_ltn": sgs(10613),
              "prazo_lft": sgs(10614),
              # bloco 5: inflação e atividade depois da flutuação
              "ipca_12m": sgs(13522, "01/01/1996", "31/12/2001"),
              "pib_crescimento_anual": sgs(7326, "01/01/1994", "31/12/2001")}

    # Fiscal, mensal, Governo Federal + Banco Central (o consolidado só existe
    # no SGS a partir de 2001-02). Sinal das NFSP: positivo = déficit.
    # R$ milhões correntes, exceto as séries "_12m_pib" (% do PIB, 12 meses).
    FISCAL = {"primario_gf": 4639, "nominal_gf": 4573, "juros_gf": 4606,
              "primario_gf_12m_pib": 5783, "nominal_gf_12m_pib": 5717,
              "juros_gf_12m_pib": 5750, "dlsp_gf": 4468,
              "pib_mensal": 4380, "igpdi_var": 190}
    fiscal = {n: sgs(c, "01/01/1991", "31/12/2006") for n, c in FISCAL.items()}
    pd.DataFrame(fiscal).to_csv(DADOS / "fiscal_mensal.csv", index_label="data")
    mensal.update({f"fiscal:{n}": v for n, v in fiscal.items()})

    for nome, s in series.items():
        s.rename(nome).to_csv(DADOS / f"{nome}_diario.csv", index_label="data")
    for nome, s in mensal.items():
        if nome.startswith("fiscal:"):
            continue
        s.rename(nome).to_csv(DADOS / f"{nome}.csv", index_label="data")

    # Painel em dias úteis brasileiros (datas da PTAX); Fed funds e EMBI
    # alinhados pela última observação disponível (sem preencher além de 5 dias).
    painel = pd.DataFrame(series)
    painel = painel.loc[series["ptax"].index].sort_index()
    painel[["fedfunds", "embi"]] = (
        pd.DataFrame(series)[["fedfunds", "embi"]].sort_index().ffill(limit=5)
        .reindex(painel.index))
    painel.to_csv(DADOS / "painel_diario.csv", index_label="data")

    with open(DADOS / "LEIAME_coleta.txt", "w") as f:
        f.write(f"Consulta em {date.today().isoformat()}\n")
        for nome, s in {**series, **mensal}.items():
            f.write(f"{nome}: {len(s)} obs, {s.index.min().date()} a "
                    f"{s.index.max().date()}\n")
    print(open(DADOS / "LEIAME_coleta.txt").read())
    print(painel.describe().T.round(2))


if __name__ == "__main__":
    main()
