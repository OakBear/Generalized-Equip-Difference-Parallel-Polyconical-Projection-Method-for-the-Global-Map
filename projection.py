"""
Generalized Equip-Difference Parallel Polyconical Projection
for the Global Map

Reference
---------
Hao Xiaoguang, "Generalized Equip-Difference Parallel Polyconical Projection
Method for the Global Map".

Overview
--------
Traditional world maps use longitude as the horizontal (x) coordinate and
latitude as the vertical (y) coordinate, which causes severe shape and area
distortion near the north and south poles.

This module implements the method proposed by Hao Xiaoguang, which:

1.  Keeps **latitude as the vertical coordinate** so that the spacing of
    parallels along the central meridian is equidistant (equal-difference).
2.  Introduces **generalized longitude** (Λ) and **generalized latitude** (Φ)
    — new coordinate variables derived from the conventional longitude (λ) and
    latitude (φ) through the polyconical mapping geometry.
3.  Uses a separate tangent cone for each parallel (polyconical), giving the
    radius of the parallel circle as  ρ(φ) = R / tan(φ).
4.  Derives explicit mathematical relations between (Φ, Λ) and (φ, λ) so
    that the resulting "plane terrestrial globe" faithfully represents land
    shapes and areas across the whole world.

Mathematical Relations
----------------------
Given Earth radius R, conventional latitude φ and longitude λ (both in
radians), the generalized latitude Φ and generalized longitude Λ are:

    Φ = φ                           (generalized latitude = conventional latitude)
    Λ = λ · sin(φ)                  (cone arc-angle at latitude φ; 0 at equator)

Inverse (Λ, Φ) → (λ, φ):

    φ = Φ
    λ = Λ / sin(Φ)    for |Φ| > 0  (undefined / continuous limit at Φ = 0)

2-D projection coordinates (x, y):

    For φ ≠ 0:
        ρ = R / tan(φ)                 (parallel radius on cone)
        x = ρ · sin(Λ)
        y = R · φ − ρ · (1 − cos(Λ))

    For φ = 0 (equator, limiting case):
        x = R · λ
        y = 0
"""

import math

import numpy as np

# Earth mean radius (km); can be overridden per instance.
EARTH_RADIUS_KM = 6371.0


class GeneralizedPolyconicalProjection:
    """
    Generalized Equip-Difference Parallel Polyconical Projection.

    Parameters
    ----------
    radius : float
        Earth radius used for scaling.  Defaults to ``EARTH_RADIUS_KM``
        (6 371 km).  Set to 1.0 for dimensionless / normalised coordinates.
    central_meridian : float
        Central meridian in **degrees** (default 0 °).  All input longitudes
        are measured relative to this meridian before projection.
    """

    def __init__(self, radius: float = EARTH_RADIUS_KM, central_meridian: float = 0.0):
        self.radius = float(radius)
        self.central_meridian = float(central_meridian)

    # ------------------------------------------------------------------
    # Coordinate transformations  (φ, λ)  ↔  (Φ, Λ)
    # ------------------------------------------------------------------

    def conventional_to_generalized(
        self,
        lon_deg,
        lat_deg,
    ):
        """
        Convert conventional (longitude, latitude) to generalized (Λ, Φ).

        The generalized latitude Φ is identical to the conventional latitude φ
        because latitude is used directly as the vertical coordinate.

        The generalized longitude Λ is the arc-angle subtended by the
        meridional span λ on the tangent cone at latitude φ:

            Λ = λ · sin(φ)

        This relationship arises from the polyconical cone geometry: the
        parallel at latitude φ lies on a cone whose apex half-angle equals
        (90° − φ), so a full 360° of longitude maps to an arc of
        2π · sin(φ) radians on the cone.

        Parameters
        ----------
        lon_deg : float or array-like
            Conventional longitude in degrees, range [−180, 180].  Values
            outside this range are accepted and handled correctly.
        lat_deg : float or array-like
            Conventional latitude in degrees, range [−90, 90].

        Returns
        -------
        gen_lon_deg : float or ndarray
            Generalized longitude Λ in degrees.
        gen_lat_deg : float or ndarray
            Generalized latitude Φ in degrees (equals ``lat_deg``).
        """
        lon = np.asarray(lon_deg, dtype=float)
        lat = np.asarray(lat_deg, dtype=float)
        lat_rad = np.radians(lat)
        # Shift longitude to be relative to the central meridian
        lon_rel = lon - self.central_meridian

        # Φ = φ
        gen_lat = lat

        # Λ = λ · sin(φ)   (degrees · dimensionless = degrees of arc-angle)
        gen_lon = lon_rel * np.sin(lat_rad)

        return gen_lon, gen_lat

    def generalized_to_conventional(
        self,
        gen_lon_deg,
        gen_lat_deg,
    ):
        """
        Convert generalized coordinates (Λ, Φ) back to conventional (λ, φ).

        Inverse of :meth:`conventional_to_generalized`:

            φ = Φ
            λ = Λ / sin(Φ)   for |Φ| > 0°

        At the equator (Φ = 0) the generalized longitude is always zero for
        any conventional longitude (all meridians touch the equator at the
        same height), so the inverse is not unique there; the method returns
        λ = 0 at the equator.

        Parameters
        ----------
        gen_lon_deg : float or array-like
            Generalized longitude Λ in degrees.
        gen_lat_deg : float or array-like
            Generalized latitude Φ in degrees.

        Returns
        -------
        lon_deg : float or ndarray
            Conventional longitude in degrees (relative to the central meridian).
        lat_deg : float or ndarray
            Conventional latitude in degrees.
        """
        gen_lon = np.asarray(gen_lon_deg, dtype=float)
        gen_lat = np.asarray(gen_lat_deg, dtype=float)
        gen_lat_rad = np.radians(gen_lat)

        # φ = Φ
        lat = gen_lat

        # λ = Λ / sin(Φ), with graceful handling of the equator
        sin_lat = np.sin(gen_lat_rad)
        lon_rel = np.where(np.abs(sin_lat) > 1e-10, gen_lon / sin_lat, 0.0)
        lon = lon_rel + self.central_meridian

        return lon, lat

    # ------------------------------------------------------------------
    # Forward projection  (φ, λ)  →  (x, y)
    # ------------------------------------------------------------------

    def project(self, lon_deg, lat_deg):
        """
        Project conventional longitude-latitude to 2-D plane coordinates.

        The equip-difference parallel polyconical formulae are:

        **Equator** (φ = 0):
            x = R · λ,   y = 0

        **All other latitudes** (φ ≠ 0):
            ρ = R / tan(φ)           — radius of the parallel on its cone
            Λ = λ · sin(φ)           — generalized longitude (radians)
            x = ρ · sin(Λ)
            y = R · φ − ρ · (1 − cos(Λ))

        Parallels are equidistant along the central meridian (λ = 0),
        where every formula reduces to  x = 0, y = R · φ.

        Parameters
        ----------
        lon_deg : float or array-like
            Longitude in degrees.
        lat_deg : float or array-like
            Latitude in degrees.

        Returns
        -------
        x : float or ndarray
            Projected x-coordinate (same units as ``self.radius``).
        y : float or ndarray
            Projected y-coordinate (same units as ``self.radius``).
        """
        lon = np.asarray(lon_deg, dtype=float)
        lat = np.asarray(lat_deg, dtype=float)

        scalar_input = lon.ndim == 0 and lat.ndim == 0
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)

        lon_rad = np.radians(lon - self.central_meridian)
        lat_rad = np.radians(lat)

        x = np.empty_like(lat_rad)
        y = np.empty_like(lat_rad)

        # Equator case
        eq = np.abs(lat_rad) < 1e-10
        x[eq] = self.radius * lon_rad[eq]
        y[eq] = 0.0

        # General case
        non_eq = ~eq
        if np.any(non_eq):
            phi = lat_rad[non_eq]
            lam = lon_rad[non_eq]
            rho = self.radius / np.tan(phi)       # ρ = R / tan(φ)
            gen_lon_rad = lam * np.sin(phi)        # Λ = λ · sin(φ)
            x[non_eq] = rho * np.sin(gen_lon_rad)
            y[non_eq] = self.radius * phi - rho * (1.0 - np.cos(gen_lon_rad))

        if scalar_input:
            return float(x[0]), float(y[0])
        return x, y

    # ------------------------------------------------------------------
    # Inverse projection  (x, y)  →  (φ, λ)
    # ------------------------------------------------------------------

    def inverse_project(self, x, y, tol: float = 1e-10, max_iter: int = 100):
        """
        Recover conventional longitude-latitude from 2-D plane coordinates.

        Solves the forward equation element-wise:

            F(φ) = R·φ − ρ·(1 − cos Λ) − y = 0,
            ρ = R/tan(φ),   Λ = arcsin(x·tan(φ)/R)

        Strategy per element:

        1. Newton-Raphson from φ₀ = y/R (exact on the central meridian).
        2. If the residual after NR exceeds *tol*, fall back to **bisection**
           on [−π/2 + ε, π/2 − ε].  Bisection is guaranteed to converge
           because F(−π/2) < 0 and F(π/2) > 0 for all finite (x, y).

        **Note on injectivity**: The polyconic projection is not globally
        injective for large longitudinal extents.  Some (x, y) pairs have
        multiple pre-images; the method always returns a valid pre-image,
        i.e. ``project(inverse(x, y)) ≈ (x, y)``.

        Parameters
        ----------
        x : float or array-like
            Projected x-coordinate.
        y : float or array-like
            Projected y-coordinate.
        tol : float
            Convergence tolerance on the residual |F(φ)|.
        max_iter : int
            Maximum iterations (shared by NR and bisection phases).

        Returns
        -------
        lon_deg : float or ndarray
            Recovered longitude in degrees.
        lat_deg : float or ndarray
            Recovered latitude in degrees.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        scalar_input = x_arr.ndim == 0 and y_arr.ndim == 0
        x_arr = np.atleast_1d(x_arr).copy()
        y_arr = np.atleast_1d(y_arr).copy()

        R = self.radius
        eps = 1e-9  # keep away from poles
        h = 1e-6    # central-difference step

        # ------------------------------------------------------------------
        # Per-element residual function
        # ------------------------------------------------------------------
        def _f_scalar(phi_val, idx):
            if abs(phi_val) < 1e-10:
                return -y_arr[idx]
            tan_phi = math.tan(phi_val)
            rho = R / tan_phi
            ratio = max(-1.0, min(1.0, x_arr[idx] * tan_phi / R))
            gen_lon = math.asin(ratio)
            return R * phi_val - rho * (1.0 - math.cos(gen_lon)) - y_arr[idx]

        lat_rad = np.empty(len(y_arr))

        for idx in range(len(y_arr)):
            # --- Phase 1: Newton-Raphson ---
            phi = float(np.clip(y_arr[idx] / R, -math.pi / 2 + eps, math.pi / 2 - eps))
            for _ in range(max_iter):
                fv = _f_scalar(phi, idx)
                if abs(fv) < tol:
                    break
                phi_p = min(phi + h, math.pi / 2 - eps)
                phi_m = max(phi - h, -math.pi / 2 + eps)
                df = (_f_scalar(phi_p, idx) - _f_scalar(phi_m, idx)) / (phi_p - phi_m)
                if abs(df) < 1e-20:
                    break
                step = max(-0.5, min(0.5, fv / df))
                phi = max(-math.pi / 2 + eps, min(math.pi / 2 - eps, phi - step))

            # --- Phase 2: Bisection fallback if NR residual is too large ---
            if abs(_f_scalar(phi, idx)) > tol:
                lo, hi = -math.pi / 2 + eps, math.pi / 2 - eps
                f_lo = _f_scalar(lo, idx)
                for _ in range(max_iter + 50):
                    mid = (lo + hi) * 0.5
                    if (hi - lo) < tol:
                        break
                    f_mid = _f_scalar(mid, idx)
                    if abs(f_mid) < tol:
                        break
                    if math.copysign(1.0, f_mid) == math.copysign(1.0, f_lo):
                        lo = mid
                        f_lo = f_mid
                    else:
                        hi = mid
                phi = (lo + hi) * 0.5

            lat_rad[idx] = phi

        # ------------------------------------------------------------------
        # Recover longitude from  x = ρ · sin(Λ)
        # ------------------------------------------------------------------
        lon_rad = np.zeros_like(lat_rad)

        non_eq = np.abs(lat_rad) > 1e-10
        eq = ~non_eq
        lon_rad[eq] = x_arr[eq] / R          # equator: x = R·λ

        if np.any(non_eq):
            phi = lat_rad[non_eq]
            rho = R / np.tan(phi)
            ratio = np.clip(x_arr[non_eq] / rho, -1.0, 1.0)
            gen_lon_rad = np.arcsin(ratio)
            lon_rad[non_eq] = gen_lon_rad / np.sin(phi)

        lon_rad += np.radians(self.central_meridian)

        if scalar_input:
            return float(np.degrees(lon_rad[0])), float(np.degrees(lat_rad[0]))
        return np.degrees(lon_rad), np.degrees(lat_rad)
