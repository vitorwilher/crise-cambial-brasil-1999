"""Padrão visual da Análise Macro para os gráficos do doutorado.

Tokens do tema CLARO da casa (decisão de 11/09/2026), copiados de
`4. Portfolio/ROI_Diagnostico/CLAUDE.md` e `social/carousel_charts.py`. Fundo claro pede a
logo NAVY (`logo-am-navy.png`, ao lado deste arquivo).

Uso:
    import sys; sys.path.insert(0, '<...>/Projetos/estilo_am')
    import estilo_am as am
    fig, axs = am.figura(2, 1, figsize=(9.5, 7))
    am.eixo(ax)                         # para cada eixo
    am.titulos(fig, 'Título', 'Subtítulo')
    am.rodape(fig, 'Fonte: ...')
    am.salvar(fig, 'saidas/grafico.png')
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# tokens
PAPER, PAPER2 = '#FFFFFF', '#F7F9FC'
NAVY, NAVY_INK = '#0A1A3C', '#14213D'
INK, INK_SOFT = '#16202E', '#3A4757'
MUTED, LINE = '#8894A5', '#E6EAF0'
BLUE, BLUE_DEEP = '#1CA0D8', '#1877F2'
BLUE_SOFT, BLUE_WASH = '#E7F4FB', '#BFE3F5'
GREEN, RED, AMBER = '#12B76A', '#E5484D', '#F5A524'

# ordem das séries: navy primeiro (série principal), azul-marca depois, âmbar como contraste
SERIES = [NAVY, BLUE, AMBER, MUTED]

LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo-am-navy.png')

plt.rcParams.update({
    'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER,
    'axes.edgecolor': LINE, 'axes.labelcolor': INK_SOFT, 'axes.titlecolor': NAVY,
    'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})


def figura(nrows=1, ncols=1, **kw):
    fig, axs = plt.subplots(nrows, ncols, **kw)
    for ax in (axs.flat if hasattr(axs, 'flat') else [axs]):
        eixo(ax)
    return fig, axs


def eixo(ax):
    """Sem moldura no topo e à direita, tick sem traço, grade horizontal suave."""
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(LINE)
    ax.tick_params(colors=MUTED, length=0, labelsize=9.5)
    ax.grid(axis='y', color=LINE, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def subtitulo_eixo(ax, texto):
    ax.set_title(texto, loc='left', fontsize=11, color=NAVY, fontweight='bold', pad=8)


def titulos(fig, titulo, subtitulo=None, x=0.06):
    _moldura(fig, fig.text(x, 0.985, titulo, ha='left', va='top', fontsize=14,
                           fontweight='bold', color=NAVY))
    if subtitulo:
        _moldura(fig, fig.text(x, 0.945, subtitulo, ha='left', va='top', fontsize=10.5,
                               color=INK_SOFT))
    _logo(fig)


def _moldura(fig, artista):
    """Guarda título, subtítulo, rodapé e logo: a versão para slide sai sem eles."""
    if not hasattr(fig, '_am_moldura'):
        fig._am_moldura = []
    fig._am_moldura.append(artista)


def rodape(fig, texto, x=0.06, y=0.012):
    _moldura(fig, fig.text(x, y, texto, ha='left', va='bottom', fontsize=7.8, color=MUTED,
                           linespacing=1.4))


def _logo(fig, largura=0.07):
    if not os.path.exists(LOGO):
        return
    img = mpimg.imread(LOGO)
    w, h = fig.get_size_inches()
    altura = largura * w / h
    ax = fig.add_axes([0.98 - largura, 0.985 - altura, largura, altura])
    ax.imshow(img)
    ax.axis('off')
    _moldura(fig, ax)


def rotulo_final(ax, x, y, texto, cor=NAVY, dy=6):
    """Rótulo direto no último ponto, em tinta de texto (a série é identificada pela cor da marca)."""
    ax.plot([x], [y], 'o', ms=5.5, color=cor, mec=PAPER, mew=1.8, zorder=5)
    ax.annotate(texto, (x, y), xytext=(6, dy), textcoords='offset points', fontsize=8.8,
                color=INK, fontweight='bold')


def salvar(fig, caminho, dpi=200):
    """Salva a figura e, se ela tem moldura, uma versão para slide em slides/ (só os painéis)."""
    fig.savefig(caminho, dpi=dpi, facecolor=PAPER)
    if getattr(fig, '_am_moldura', None):
        for a in fig._am_moldura:
            a.set_visible(False)
        destino = Path(caminho).parent / 'slides'
        destino.mkdir(exist_ok=True)
        fig.savefig(destino / Path(caminho).name, dpi=dpi, facecolor=PAPER,
                    bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)


def br(v, casas=1):
    """Número no formato brasileiro."""
    return f'{v:,.{casas}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def eixo_br(ax, casas=1, sufixo='', eixo='y'):
    """Rótulos do eixo com vírgula decimal."""
    f = lambda v, _: br(v, casas) + sufixo
    (ax.yaxis if eixo == 'y' else ax.xaxis).set_major_formatter(f)


_MESES = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']


def eixo_datas(ax, meses=None):
    """Datas em português: só o ano (padrão) ou 'mmm/aa' nos meses pedidos."""
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    if meses is None:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=meses))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: (lambda t: f'{_MESES[t.month - 1]}/{t:%y}')(mdates.num2date(x))))
