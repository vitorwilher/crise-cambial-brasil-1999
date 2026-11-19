#!/usr/bin/env bash
# Reproduz todo o trabalho a partir de dados/ vazia, na ordem do roteiro.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY src/01_coleta.py               # dados: SGS, FRED, Ipeadata
$PY src/02_painel_semanal.py       # painel semanal e prêmio implícito nos juros
$PY src/10_fiscal.py               # blocos 1-2: política fiscal
$PY src/20_divida_prazo_composicao.py  # bloco 2: prazo e indexação da dívida
$PY src/30_markov_juros.py         # blocos 3-4: regimes nos juros (2ª geração)
$PY src/31_markov_robustez.py      # robustez e bootstrap da LR (~5 min)
$PY src/32_pressao_cambial.py      # bloco 3: índice de pressão cambial
$PY src/40_contrafactual.py        # bloco 6: contrafactual (+ base do app)
quarto render slides/apresentacao/slides.qmd
