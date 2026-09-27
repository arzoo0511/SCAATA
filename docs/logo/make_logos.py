import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, FancyBboxPatch, Circle, Polygon
from matplotlib.transforms import Affine2D

TAN, MAROON, ALAB, LBLUE, MID = "#D8BA98", "#7F0303", "#EFE8DF", "#96C0CE", "#0F414A"
OUT = "docs/logo/"

def canvas(w_px, h_px, bg):
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w_px / h_px); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor(bg)
    return fig, ax

def face_transform(ax, origin, x_axis, y_axis):
    m = np.array([[x_axis[0], y_axis[0], origin[0]], [x_axis[1], y_axis[1], origin[1]], [0, 0, 1]])
    return Affine2D(m) + ax.transData

# ---- letters in a unit square, as filled shapes so the face skew applies to strokes too ----
T = 0.20
def letter_S():
    rc = 0.145
    return [Wedge((0.5, 0.5 + rc), rc + T / 2, 20, 270, width=T),
            Wedge((0.5, 0.5 - rc), rc + T / 2, -160, 90, width=T)]
def letter_C():
    return [Wedge((0.5, 0.5), 0.36, 48, 312, width=T)]
def letter_A():
    return [Wedge((0.5, 0.52), 0.33, 0, 180, width=T),
            Rectangle((0.17, 0.13), T, 0.39), Rectangle((0.83 - T, 0.13), T, 0.39),
            Rectangle((0.17 + T, 0.30), 0.66 - 2 * T, 0.13)]

def cube(ax, cx, cy, R, colors=(TAN, ALAB, LBLUE), letter_color=MID):
    c30 = np.cos(np.pi / 6)
    C0 = np.array([cx, cy]); T_ = C0 + [0, R]; B = C0 - [0, R]
    UL, UR = C0 + [-R * c30, R / 2], C0 + [R * c30, R / 2]
    LL, LR = C0 + [-R * c30, -R / 2], C0 + [R * c30, -R / 2]
    faces = [  # (origin, x-axis, y-axis, letter builder, face colour)
        (UL, C0 - UL, T_ - UL, letter_S, colors[0]),   # top
        (LL, B - LL, UL - LL, letter_C, colors[1]),    # left
        (B, LR - B, C0 - B, letter_A, colors[2]),      # right
    ]
    for origin, xa, ya, build, col in faces:
        tr = face_transform(ax, origin, xa, ya)
        bg = FancyBboxPatch((0.07, 0.07), 0.86, 0.86, boxstyle="round,pad=0,rounding_size=0.10", color=col)
        bg.set_transform(tr); ax.add_patch(bg)
        for p in build():
            p.set_color(letter_color); p.set_transform(tr); ax.add_patch(p)

def spaced_text(ax, x, y, text, size, color, font, weight="normal", tracking=0.0, ha="left"):
    t = ax.text(x, y, text, fontsize=size, color=color, family=font, weight=weight, va="center", ha=ha)
    return t

def h_mark(ax, cx, cy, size, ink=MID, dot=MAROON, bg=ALAB):
    """An H whose crossbar is a ring holding a dot: the letter looking at itself."""
    s = size
    pw, ph = 0.15 * s, 0.66 * s                     # pillars
    for x in (cx - 0.30 * s, cx + 0.30 * s - pw):
        ax.add_patch(FancyBboxPatch((x, cy - ph / 2), pw, ph, boxstyle=f"round,pad=0,rounding_size={pw/2}", color=ink))
    ro, rw = 0.23 * s, 0.12 * s                      # ring spanning the gap, fused into both pillars
    ax.add_patch(Wedge((cx, cy), ro, 0, 360, width=rw, color=ink))
    ax.add_patch(Circle((cx, cy), 0.075 * s, color=dot))

def hanko(ax, x, y, size, color=MAROON, fg=ALAB):
    ax.add_patch(FancyBboxPatch((x, y), size, size * 1.9, boxstyle=f"round,pad=0,rounding_size={size*0.12}", color=color))
    ax.text(x + size / 2, y + size * 0.95, "反" + chr(10) + "省", fontsize=size * 330, color=fg, family="Yu Gothic",
            weight="bold", va="center", ha="center", linespacing=1.0)

# ===== Concept A: SCAATA letter cube =====
fig, ax = canvas(1024, 1024, MID)
cube(ax, 0.5, 0.5, 0.36)
fig.savefig(OUT + "scaata_cube_icon.png", facecolor=MID); fig.savefig(OUT + "scaata_cube_icon.svg", facecolor=MID); plt.close(fig)

fig, ax = canvas(1800, 600, MID)
cube(ax, 0.50, 0.5, 0.34)
ax.text(1.02, 0.58, "S C A A T A", fontsize=112, color=ALAB, family="Franklin Gothic Heavy", va="center")
ax.text(1.04, 0.30, "SELF-CRITIQUING TRADING AGENT", fontsize=26, color=TAN, family="Bahnschrift", va="center")
fig.savefig(OUT + "scaata_cube_wordmark.png", facecolor=MID); fig.savefig(OUT + "scaata_cube_wordmark.svg", facecolor=MID); plt.close(fig)

def text_right_edge(fig, ax, txt):
    fig.canvas.draw()
    bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
    return ax.transData.inverted().transform((bb.x1, bb.y1)), ax.transData.inverted().transform((bb.x0, bb.y0))

# ===== Concept A wordmark, tightened to its content =====
fig, ax = canvas(1240, 560, MID)
cube(ax, 0.47, 0.5, 0.33)
name = ax.text(0.95, 0.57, "SCAATA", fontsize=118, color=ALAB, family="Franklin Gothic Heavy", va="center")
ax.text(0.965, 0.33, "SELF-CRITIQUING TRADING AGENT", fontsize=25, color=TAN, family="Bahnschrift", va="center")
fig.savefig(OUT + "scaata_cube_wordmark.png", facecolor=MID); fig.savefig(OUT + "scaata_cube_wordmark.svg", facecolor=MID); plt.close(fig)

# ===== Concept B: HANSEI =====
fig, ax = canvas(1024, 1024, ALAB)
h_mark(ax, 0.5, 0.5, 1.0)
fig.savefig(OUT + "hansei_icon.png", facecolor=ALAB); fig.savefig(OUT + "hansei_icon.svg", facecolor=ALAB); plt.close(fig)

fig, ax = canvas(1490, 560, ALAB)
h_mark(ax, 0.47, 0.5, 0.82)
name = ax.text(0.93, 0.58, "HANSEI", fontsize=150, color=MID, family="Franklin Gothic Heavy", va="center")
tag = ax.text(0.945, 0.33, "SELF-CRITIQUING TRADING AGENT", fontsize=27, color=MAROON, family="Bahnschrift", va="center")
(x1, y1), (x0, y0) = text_right_edge(fig, ax, name)
seal = 0.13
hanko(ax, x1 + 0.05, y1 - seal * 1.9, seal)
fig.savefig(OUT + "hansei_wordmark.png", facecolor=ALAB); fig.savefig(OUT + "hansei_wordmark.svg", facecolor=ALAB); plt.close(fig)
print("ok")
