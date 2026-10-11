"""Estilo común de las gráficas (paleta validada para daltonismo, ejes discretos)."""
import matplotlib.pyplot as plt

SUPERFICIE = "#fcfcfb"
TEXTO_PRIMARIO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
REJILLA = "#e4e3df"

# Paleta categórica en orden fijo: azul, naranja, aguamarina
COLORES_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
# Rampa secuencial azul (claro -> oscuro) para magnitudes y niveles ordenados
AZUL_SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COLORES_NIVELES = {"bajo": "#86b6ef", "medio": "#2a78d6", "alto": "#104281"}


def estilo_base():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "axes.edgecolor": REJILLA,
        "axes.labelcolor": TEXTO_SECUNDARIO,
        "axes.titlecolor": TEXTO_PRIMARIO,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_SECUNDARIO,
        "ytick.color": TEXTO_SECUNDARIO,
        "text.color": TEXTO_PRIMARIO,
        "font.size": 10,
        "legend.labelcolor": TEXTO_PRIMARIO,
    })
