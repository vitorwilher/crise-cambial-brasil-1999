# Brasil, 1999: por que a banda cambial colapsou?

Trabalho final de **Dívida Soberana e Câmbio** (Doutorado FGV EPGE).
Pergunta: por que o regime de bandas cambiais entrou em colapso em janeiro de 1999
e qual foi o papel das reservas, dos juros e da política fiscal?

## Reproduzir

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./rodar_tudo.sh
```

Os dados vêm das APIs públicas do BCB (SGS), FRED e Ipeadata; os gráficos e tabelas
vão para `saidas/`. Fontes, códigos das séries e o mapa roteiro → scripts estão em
[CLAUDE.md](CLAUDE.md).

## Slides

`slides/apresentacao/slides.qmd` (Quarto revealjs, tema da Análise Macro). Depois de
rodar `./rodar_tudo.sh`: `quarto render slides/apresentacao/slides.qmd`.
