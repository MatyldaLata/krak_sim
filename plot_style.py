"""Wspólny styl wykresów: jedna paleta, czytelne fonty, wymagania EuRoC rysowane zawsze tak samo.

Zasady:
  * jedna oś Y na wykres (zamiast twinx -> dwa panele ze wspólną osią X),
  * kolory serii w stałej kolejności (SERIES), kolory statusu tylko dla wymagań,
  * obszar niedozwolony zacieniowany na czerwono, limit opisany wprost na wykresie,
  * punkt nominalny zawsze zaznaczony i podpisany wartością.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#fcfcfb"

DPI = 200


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 12, "axes.labelcolor": INK2,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "xtick.color": INK2, "ytick.color": INK2,
        "legend.fontsize": 11, "legend.frameon": False,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 2.0, "lines.markersize": 7,
        "figure.titlesize": 16, "figure.titleweight": "bold",
        "axes.formatter.use_locale": False,
    })


def forbidden_above(ax, y, label, color=CRIT):
    """Zacieniuj obszar powyżej limitu i podpisz limit przy prawej krawędzi."""
    ax.axhspan(y, ax.get_ylim()[1] if ax.get_ylim()[1] > y else y * 1.5, color=color, alpha=0.12, lw=0)
    ax.axhline(y, color=color, lw=1.5, ls="--")
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", va="bottom", color=color, fontsize=10.5)


def forbidden_below(ax, y, label, color=CRIT):
    lo = min(ax.get_ylim()[0], y)
    ax.axhspan(lo, y, color=color, alpha=0.12, lw=0)
    ax.axhline(y, color=color, lw=1.5, ls="--")
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, -4),
                textcoords="offset points", ha="right", va="top", color=color, fontsize=10.5)


def allowed_band(ax, lo, hi, label):
    ax.axhspan(lo, hi, color=GOOD, alpha=0.12, lw=0)
    for y in (lo, hi):
        ax.axhline(y, color=GOOD, lw=1.2, ls="--")
    ax.annotate(label, xy=(1.0, hi), xycoords=("axes fraction", "data"), xytext=(-4, -4),
                textcoords="offset points", ha="right", va="top", color=GOOD, fontsize=10.5)


def target_line(ax, y, label, vertical=False):
    if vertical:
        ax.axvline(y, color=INK, lw=1.3, ls=":")
        ax.annotate(label, xy=(y, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
                    textcoords="offset points", ha="left", va="top", color=INK, fontsize=10.5)
    else:
        ax.axhline(y, color=INK, lw=1.3, ls=":")
        ax.annotate(label, xy=(0.0, y), xycoords=("axes fraction", "data"), xytext=(4, 4),
                    textcoords="offset points", ha="left", va="bottom", color=INK, fontsize=10.5)


def mark_nominal(ax, x, y, text, color=INK, dx=8, dy=8):
    ax.plot([x], [y], marker="o", ms=9, mfc=SURFACE, mec=color, mew=2, zorder=5)
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", fontsize=10.5,
                color=color, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))


def subtitle(fig, text):
    fig.text(0.01, 0.955, text, fontsize=11.5, color=INK2, ha="left", va="top")


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
