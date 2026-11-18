"""Painel semanal e a variável do Markov-switching.

Decomposição (Garcia e Didier, 2003), em p.p. ao ano:

    Selic − Fed funds = risco-país (EMBI) + depreciação esperada + prêmio cambial

Sob uma banda crível, a depreciação esperada é o ritmo de deslizamento da banda.
Aproximamos esse ritmo pela depreciação observada da PTAX nas 13 semanas
anteriores, anualizada. O resíduo

    premio_cambial = Selic − Fed funds − EMBI/100 − deslize

mede quanto o mercado cobrava ALÉM do deslize: a expectativa de uma
maxidesvalorização (salto da banda) mais o prêmio de risco cambial.
É a variável cujos regimes o Markov-switching procura (Jeanne e Masson, 2000).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"

d = pd.read_csv(DADOS / "painel_diario.csv", index_col=0, parse_dates=True)

sem = pd.DataFrame({
    "ptax": d["ptax"].resample("W-FRI").last(),
    "selic": d["selic"].resample("W-FRI").mean(),
    "fedfunds": d["fedfunds"].resample("W-FRI").mean(),
    "embi": d["embi"].resample("W-FRI").mean(),
})

# Deslize da banda: depreciação anualizada das 13 semanas anteriores
sem["deslize"] = ((sem["ptax"] / sem["ptax"].shift(13)) ** (52 / 13) - 1) * 100
sem["dif_juros"] = sem["selic"] - sem["fedfunds"]
sem["premio_cambial"] = sem["dif_juros"] - sem["embi"] / 100 - sem["deslize"]

# Reservas mensais (posição de fim de mês): cada semana recebe a posição do mês
# ANTERIOR, que já era conhecida, para não usar informação do futuro.
res = pd.read_csv(DADOS / "reservas_mensal.csv", index_col=0, parse_dates=True)
res = res["reservas_mensal"]
mes_ant = (sem.index.to_period("M") - 1).to_timestamp()
sem["reservas"] = res.reindex(mes_ant).values
sem["log_reservas"] = np.log(sem["reservas"])
sem["var_reservas_3m"] = (res.pct_change(3) * 100).reindex(mes_ant).values

# Prazo médio de emissão das LTN: posição do mês anterior (meses sem emissão
# ficam com o último valor conhecido, no máximo 2 meses).
pz = pd.read_csv(DADOS / "prazo_ltn.csv", index_col=0, parse_dates=True)["prazo_ltn"]
pz = pz.asfreq("MS").ffill(limit=2)
sem["prazo_ltn"] = pz.reindex(mes_ant).values

sem.to_csv(DADOS / "painel_semanal.csv", index_label="semana")
print(sem.loc["1995-03":"1999-01"].describe().T.round(2))
print(sem.loc["1997-10-17":"1997-11-28", ["selic", "embi", "deslize", "premio_cambial"]].round(2))
print(sem.loc["1998-08-14":"1998-10-02", ["selic", "embi", "deslize", "premio_cambial", "reservas"]].round(2))
