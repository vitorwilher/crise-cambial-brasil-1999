#!/usr/bin/env bash
# Publica o app (apresentação + simulador) no shinyapps.io, conta "vitorwilher".
# Pré-requisito: ./rodar_tudo.sh já rodado (gera saidas/, app/dados/ e os slides).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p app/www
cp slides/apresentacao/slides.html app/www/slides.html
rsconnect deploy shiny app -n vitorwilher --title crise-cambial-brasil-1999
