# CLAUDE.md — Trabalho Final: Brasil, crise cambial de 1999

## Objetivo

Trabalho final (em dupla) de Dívida Soberana e Câmbio (FGV EPGE).
Pergunta do enunciado: **por que o regime de bandas cambiais entrou em colapso em
janeiro de 1999 e qual foi o papel das reservas, dos juros e da política fiscal?**

Núcleo empírico: **Markov-switching** sobre a expectativa de desvalorização embutida
nos juros (diferencial Selic − Fed funds − risco-país), na linha de Jeanne e Masson
(2000): o regime de bandas viveu um equilíbrio "calmo" e um "de ataque"? O que move a
probabilidade de troca (reservas, choques externos de Ásia/Rússia, fiscal)?

Enunciado: `referencias/enunciado_trabalho_final.pdf` (roteiro de 6 itens: contexto,
vulnerabilidades, dinâmica, ≥ 2 modelos do curso, resposta de política, contrafactual).

## Fonte dos dados

| Série | Fonte | Código | Unidade |
|---|---|---|---|
| Câmbio R$/US$ PTAX venda, diário | SGS/BCB | 1 | R$/US$ |
| Selic efetiva anualizada (base 252), diária | SGS/BCB | 1178 | % a.a. |
| Reservas internacionais, liquidez, diárias | SGS/BCB | 13621 | US$ milhões |
| Fed funds efetivo, diário | FRED | DFF | % a.a. |
| EMBI+ Risco-Brasil, diário | Ipeadata | JPM366_EMBI366 | pontos-base |
| Reservas, liquidez, mensal | SGS/BCB | 3546 | US$ milhões |
| Prazo médio de emissão LTN / LFT | SGS/BCB | 10613 / 10614 | meses (unidade a confirmar) |
| NFSP GF+BC mensal: primário / nominal / juros | SGS/BCB | 4639 / 4573 / 4606 | R$ milhões (+ = déficit) |
| NFSP GF+BC 12m % PIB: primário / nominal / juros | SGS/BCB | 5783 / 5717 / 5750 | % PIB |
| Dívida líquida GF+BC | SGS/BCB | 4468 | R$ milhões |
| PIB mensal corrente / IGP-DI | SGS/BCB | 4380 / 190 | R$ milhões / % a.m. |
| Composição da dívida por indexador (dez/1994–99) | Relatórios Anuais do BC | transcrito | % |

⚠️ **Fiscal só existe para Governo Federal + BC antes de 2001** no SGS/Ipeadata
(o consolidado começa em 2001-02). O consolidado anual está nos Relatórios Anuais.

## Mapa do roteiro → scripts

| Bloco do enunciado | Script | Estado |
|---|---|---|
| 1. Contexto pré-crise | `10_fiscal.py`, `20_divida_prazo_composicao.py` | fiscal e dívida feitos; falta posição externa |
| 2. Vulnerabilidades | `10_fiscal.py` (hiato fiscal), `20_...` (prazo/indexação) | falta reservas × dívida externa de curto prazo |
| 3. Dinâmica da crise | `30_markov_juros.py`, `32_pressao_cambial.py` | falta cronologia documentada |
| 4. Interpretação teórica | `30`/`31` (2ª geração); `10` (1ª geração) | falta teste explícito de 1ª geração |
| 5. Resposta de política | — | a fazer (FMI, metas, custo) |
| 6. Contrafactual | `40_contrafactual.py` + simulador em `app/` | feito (equilíbrio parcial) |

Rodar tudo: `./rodar_tudo.sh`. Publicar o app no shinyapps.io: `./publicar_app.sh`.

- SGS: `api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados` (janelas de até 10 anos)
- Ipeadata: `ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='...')`
- FRED: `fred.stlouisfed.org/graph/fredgraph.csv?id=DFF`
- Período de coleta: 1994-07 a 2000-12. Amostra do Markov-switching: regime de
  bandas (mar/1995) até a flutuação (15/01/1999).

⚠️ **Toda série citada neste projeto vem destas fontes.** Datas de evento (Ásia,
Rússia, acordo com o FMI, moratória de Minas, banda diagonal endógena, flutuação)
precisam de fonte documental citada no texto.

## Convenções

| Pasta | O quê |
|---|---|
| `dados/` | o que se baixa da fonte, **bruto, nunca editado à mão** (+ `painel_*.csv` gerados) |
| `saidas/` | gráficos, tabelas, resultados de estimação |
| `src/` | scripts numerados na ordem de execução (`01_coleta.py`, `02_...`) |
| `referencias/` | enunciado e artigos |

Gráficos no padrão `estilo_am` das notas de aula. Tabelas com cabeçalho cinza.

## Regras

**1. Nunca inventar número.** Todo valor citado vem da fonte, com a data da consulta.
Se o dado não cobre o período pedido, diga o que há e o que falta; não estime.

**2. Acumulado é composto, não soma** (inflação, depreciação, juros).

**3. Conferir o dado, não só o código.** Depois de rodar, confira linhas, primeiro e
último período, lacunas e valores de controle (ex.: PTAX ≈ 1,21 no início de jan/1999).

**4. Nenhuma afirmação sem lastro na série ou em referência citada.**

**5. Markov-switching:** reportar sempre log-verossimilhança, parâmetros por regime,
duração esperada, probabilidades suavizadas em gráfico, e testar robustez a
frequência (diária × semanal), amostra e número de regimes. Não declarar "2 regimes"
sem comparar com o modelo linear.

## Como saber que deu certo

- `src/01_coleta.py` roda do zero com `dados/` vazia
- Valores de controle batem com a fonte
- As probabilidades suavizadas são reproduzíveis com a mesma semente/chutes iniciais

## Stack

- Python ≥ 3.10 em `.venv` (`requests`, `pandas`, `numpy`, `matplotlib`, `statsmodels`)
- Markov-switching: `statsmodels.tsa.regime_switching.MarkovRegression`
  (aceita probabilidades de transição variáveis via `exog_tvtp`)
