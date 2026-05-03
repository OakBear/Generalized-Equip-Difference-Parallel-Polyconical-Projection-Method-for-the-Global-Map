"""
Visualization utilities for the Generalized Equip-Difference Parallel
Polyconical Projection.

This module provides functions to draw the graticule (grid of parallels and
meridians) and a simplified world outline on the projected plane, producing
the "plane terrestrial globe" described by Hao Xiaoguang.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

from projection import GeneralizedPolyconicalProjection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _project_line(proj, lons, lats, n_interp: int = 200):
    """
    Project a sequence of (lon, lat) waypoints to (x, y), interpolating
    *n_interp* points between consecutive waypoints for smooth curves.
    Returns (x_arr, y_arr) as 1-D numpy arrays, with NaN breaks where
    the input already contains NaN waypoints.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    xs, ys = [], []
    for i in range(len(lons) - 1):
        if np.isnan(lons[i]) or np.isnan(lats[i]) or \
           np.isnan(lons[i + 1]) or np.isnan(lats[i + 1]):
            xs.append(np.nan)
            ys.append(np.nan)
            continue
        seg_lons = np.linspace(lons[i], lons[i + 1], n_interp)
        seg_lats = np.linspace(lats[i], lats[i + 1], n_interp)
        sx, sy = proj.project(seg_lons, seg_lats)
        xs.append(sx)
        ys.append(sy)

    if not xs:
        return np.array([]), np.array([])

    x_arr = np.concatenate([np.atleast_1d(v) for v in xs])
    y_arr = np.concatenate([np.atleast_1d(v) for v in ys])
    return x_arr, y_arr


# ---------------------------------------------------------------------------
# Public drawing functions
# ---------------------------------------------------------------------------

def draw_graticule(
    ax,
    proj: GeneralizedPolyconicalProjection,
    lat_step: float = 30.0,
    lon_step: float = 30.0,
    n_points: int = 500,
    parallel_kw: dict = None,
    meridian_kw: dict = None,
):
    """
    Draw the graticule (grid of parallels and meridians) on *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    proj : GeneralizedPolyconicalProjection
        Projection instance to use.
    lat_step : float
        Latitude interval between drawn parallels (degrees).
    lon_step : float
        Longitude interval between drawn meridians (degrees).
    n_points : int
        Number of sample points along each line.
    parallel_kw : dict, optional
        Extra keyword arguments forwarded to ``ax.plot`` for parallels.
    meridian_kw : dict, optional
        Extra keyword arguments forwarded to ``ax.plot`` for meridians.
    """
    if parallel_kw is None:
        parallel_kw = {}
    if meridian_kw is None:
        meridian_kw = {}

    p_defaults = dict(color="steelblue", linewidth=0.5, alpha=0.6, zorder=1)
    p_defaults.update(parallel_kw)

    m_defaults = dict(color="steelblue", linewidth=0.5, alpha=0.6, zorder=1)
    m_defaults.update(meridian_kw)

    lons_full = np.linspace(-180.0, 180.0, n_points)
    lats_full = np.linspace(-90.0, 90.0, n_points)

    # Draw parallels
    for lat in np.arange(-90.0, 90.0 + lat_step / 2, lat_step):
        lats = np.full(n_points, lat)
        x, y = proj.project(lons_full, lats)
        ax.plot(x, y, **p_defaults)

    # Draw meridians
    for lon in np.arange(-180.0, 180.0 + lon_step / 2, lon_step):
        lons = np.full(n_points, lon)
        x, y = proj.project(lons, lats_full)
        ax.plot(x, y, **m_defaults)


def draw_tissot_indicatrices(
    ax,
    proj: GeneralizedPolyconicalProjection,
    lat_step: float = 30.0,
    lon_step: float = 60.0,
    radius_deg: float = 8.0,
    n_circle: int = 72,
    circle_kw: dict = None,
):
    """
    Draw Tissot's indicatrices to illustrate distortion.

    Each indicatrix is a small circle on the sphere projected into the plane.
    Perfect circles indicate no distortion; ellipses indicate shear or scaling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    proj : GeneralizedPolyconicalProjection
        Projection instance to use.
    lat_step, lon_step : float
        Spacing between indicatrix centres (degrees).
    radius_deg : float
        Angular radius of each indicatrix circle (degrees).
    n_circle : int
        Number of points per circle.
    circle_kw : dict, optional
        Extra keyword arguments forwarded to ``ax.fill``.
    """
    if circle_kw is None:
        circle_kw = {}
    c_defaults = dict(color="tomato", alpha=0.35, linewidth=0.6,
                      edgecolor="darkred", zorder=3)
    c_defaults.update(circle_kw)

    theta = np.linspace(0, 2 * np.pi, n_circle)

    for lat_c in np.arange(-60.0, 61.0, lat_step):
        for lon_c in np.arange(-150.0, 151.0, lon_step):
            # Small circle around (lon_c, lat_c)
            d_lat = radius_deg * np.cos(theta)
            d_lon = radius_deg / max(np.cos(np.radians(lat_c)), 1e-3) * np.sin(theta)
            circ_lons = lon_c + d_lon
            circ_lats = lat_c + d_lat
            circ_lats = np.clip(circ_lats, -89.9, 89.9)
            cx, cy = proj.project(circ_lons, circ_lats)
            ax.fill(cx, cy, **c_defaults)


def draw_global_map(
    proj: GeneralizedPolyconicalProjection = None,
    figsize=(14, 9),
    lat_step: float = 30.0,
    lon_step: float = 30.0,
    show_tissot: bool = False,
    title: str = "Generalized Equip-Difference Parallel Polyconical Projection\n"
                  "(Plane Terrestrial Globe)",
    savefig: str = None,
):
    """
    Draw and optionally save the global map using the polyconical projection.

    The map shows:
    * A colour-coded graticule with labelled parallels and meridians.
    * Axis annotations explaining the generalized coordinate system.
    * Optionally, Tissot's indicatrices to visualise distortion.

    Parameters
    ----------
    proj : GeneralizedPolyconicalProjection, optional
        Projection instance.  If *None*, a default instance with normalised
        radius (R = 1) and central meridian at 0° is created.
    figsize : tuple
        Figure size in inches.
    lat_step, lon_step : float
        Grid spacing in degrees.
    show_tissot : bool
        Whether to overlay Tissot indicatrices.
    title : str
        Figure title.
    savefig : str, optional
        File path to save the figure (e.g. ``"global_map.png"``).
        If *None*, the figure is not saved automatically.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    if proj is None:
        proj = GeneralizedPolyconicalProjection(radius=1.0, central_meridian=0.0)

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------
    # Graticule
    # ------------------------------------------------------------------
    draw_graticule(ax, proj, lat_step=lat_step, lon_step=lon_step)

    # ------------------------------------------------------------------
    # Highlight equator and central meridian
    # ------------------------------------------------------------------
    n = 500
    lons_full = np.linspace(-180.0, 180.0, n)
    lats_full = np.linspace(-90.0, 90.0, n)

    # Equator
    eq_x, eq_y = proj.project(lons_full, np.zeros(n))
    ax.plot(eq_x, eq_y, color="darkorange", linewidth=1.4, zorder=2,
            label="Equator (φ = 0°)")

    # Central meridian
    cm_x, cm_y = proj.project(
        np.full(n, proj.central_meridian), lats_full
    )
    ax.plot(cm_x, cm_y, color="darkred", linewidth=1.4, zorder=2,
            label=f"Central meridian (λ = {proj.central_meridian:.0f}°)")

    # ------------------------------------------------------------------
    # Pole markers
    # ------------------------------------------------------------------
    np_x, np_y = proj.project(0.0, 90.0)
    sp_x, sp_y = proj.project(0.0, -90.0)
    ax.plot(np_x, np_y, "^", color="navy", markersize=8, zorder=5, label="North Pole")
    ax.plot(sp_x, sp_y, "v", color="navy", markersize=8, zorder=5, label="South Pole")

    # ------------------------------------------------------------------
    # Tissot indicatrices (optional)
    # ------------------------------------------------------------------
    if show_tissot:
        draw_tissot_indicatrices(ax, proj)

    # ------------------------------------------------------------------
    # Latitude and longitude labels along axes
    # ------------------------------------------------------------------
    R = proj.radius
    cm_x0, _ = proj.project(proj.central_meridian, 0.0)

    for lat_label in np.arange(-90, 91, lat_step):
        lx, ly = proj.project(proj.central_meridian, lat_label)
        ax.annotate(
            f"{int(lat_label):+d}°",
            xy=(lx, ly),
            xytext=(-38, 0),
            textcoords="offset points",
            fontsize=7,
            color="darkred",
            va="center",
            ha="right",
        )

    eq_y_val = 0.0
    for lon_label in np.arange(-180, 181, lon_step):
        lx, ly = proj.project(lon_label, 0.0)
        ax.annotate(
            f"{int(lon_label):+d}°",
            xy=(lx, ly),
            xytext=(0, -14),
            textcoords="offset points",
            fontsize=7,
            color="darkorange",
            va="top",
            ha="center",
        )

    # ------------------------------------------------------------------
    # Axis cosmetics
    # ------------------------------------------------------------------
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel(
        "x  ← generalized longitude Λ = λ · sin(φ)  [units of R]",
        fontsize=9,
    )
    ax.set_ylabel(
        "y  ← generalized latitude Φ = φ  [units of R]",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(False)

    # Bounding box annotation
    x_eq_max = R * np.radians(180.0)
    ax.set_xlim(-x_eq_max * 1.05, x_eq_max * 1.05)
    ax.set_ylim(-R * np.radians(90.0) * 1.12, R * np.radians(90.0) * 1.12)

    # Text box with key equations
    eq_text = (
        "Key relations:\n"
        r"$\Phi = \varphi$" + "\n"
        r"$\Lambda = \lambda \cdot \sin\varphi$" + "\n"
        r"$\rho = R / \tan\varphi$" + "\n"
        r"$x = \rho\,\sin\Lambda$" + "\n"
        r"$y = R\varphi - \rho(1-\cos\Lambda)$"
    )
    ax.text(
        0.01, 0.97, eq_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="goldenrod", alpha=0.9),
    )

    fig.tight_layout()

    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")

    return fig, ax
