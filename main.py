"""
Main script — generate the "plane terrestrial globe" using the Generalized
Equip-Difference Parallel Polyconical Projection.

Usage
-----
    python main.py [--output <path>] [--radius <float>] [--tissot]

The script produces a PNG image of the global map and, optionally, prints a
table of sample coordinate conversions to stdout.
"""

import argparse
import sys

import numpy as np

from projection import GeneralizedPolyconicalProjection, EARTH_RADIUS_KM
from visualization import draw_global_map


# ---------------------------------------------------------------------------
# Sample coordinate table
# ---------------------------------------------------------------------------

def print_coordinate_table(proj: GeneralizedPolyconicalProjection) -> None:
    """Print a table of conventional → generalized → projected coordinates."""
    header = (
        f"{'lon (λ)':>10} {'lat (φ)':>8} │ "
        f"{'Λ (gen_lon)':>12} {'Φ (gen_lat)':>12} │ "
        f"{'x':>10} {'y':>10}"
    )
    sep = "─" * len(header)
    print("\nSample coordinate conversions")
    print(sep)
    print(header)
    print(sep)

    samples = [
        (0, 0), (90, 0), (-180, 0), (180, 0),
        (0, 30), (60, 30), (-120, 30),
        (0, 60), (90, 60),
        (0, 90), (0, -90),
        (45, 45), (-135, -45),
    ]
    for lon, lat in samples:
        gen_lon, gen_lat = proj.conventional_to_generalized(lon, lat)
        x, y = proj.project(lon, lat)
        print(
            f"{lon:>10.1f} {lat:>8.1f} │ "
            f"{gen_lon:>12.4f} {gen_lat:>12.4f} │ "
            f"{x:>10.4f} {y:>10.4f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------

def verify_round_trip(proj: GeneralizedPolyconicalProjection) -> None:
    """
    Verify the inverse projection by checking *image consistency*:
    ``forward(inverse(x, y)) ≈ (x, y)``.

    The polyconic projection is not globally injective (multiple geographic
    coordinates can map to the same plane point), so the inverse cannot in
    general recover the original (lon, lat).  The correct invariant is that
    re-projecting the recovered coordinates reproduces the original projected
    coordinates.
    """
    rng = np.random.default_rng(42)
    lons = rng.uniform(-90.0, 90.0, 200)
    lats = rng.uniform(-80.0, 80.0, 200)

    x, y = proj.project(lons, lats)
    lons_rec, lats_rec = proj.inverse_project(x, y)
    x_back, y_back = proj.project(lons_rec, lats_rec)

    x_err = np.max(np.abs(x_back - x))
    y_err = np.max(np.abs(y_back - y))
    print(f"\nInverse projection image-consistency check (200 random points):")
    print(f"  Max x residual : {x_err:.2e}")
    print(f"  Max y residual : {y_err:.2e}")
    if x_err < 1e-4 and y_err < 1e-4:
        print("  ✓  Image consistency within tolerance.")
    else:
        print("  ✗  Image consistency error exceeds tolerance — check implementation.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the Generalized Equip-Difference Parallel "
                    "Polyconical Projection global map."
    )
    parser.add_argument(
        "--output", "-o",
        default="global_map.png",
        help="Output image file (default: global_map.png)",
    )
    parser.add_argument(
        "--radius", "-r",
        type=float,
        default=1.0,
        help="Earth radius for projection (default: 1.0, dimensionless)",
    )
    parser.add_argument(
        "--central-meridian", "-c",
        type=float,
        default=0.0,
        dest="central_meridian",
        help="Central meridian in degrees (default: 0)",
    )
    parser.add_argument(
        "--tissot",
        action="store_true",
        help="Overlay Tissot indicatrices to visualise distortion",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print sample coordinate conversion table to stdout",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    proj = GeneralizedPolyconicalProjection(
        radius=args.radius,
        central_meridian=args.central_meridian,
    )

    if args.table:
        print_coordinate_table(proj)
        verify_round_trip(proj)

    print(f"\nGenerating map → {args.output} …")
    fig, ax = draw_global_map(
        proj=proj,
        show_tissot=args.tissot,
        savefig=args.output,
    )
    print("Done.")
    return fig, ax


if __name__ == "__main__":
    main()
