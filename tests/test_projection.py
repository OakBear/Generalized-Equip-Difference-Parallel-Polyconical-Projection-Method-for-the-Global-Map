"""
Unit tests for the Generalized Equip-Difference Parallel Polyconical
Projection implementation.

Run with:
    pytest tests/test_projection.py -v
"""

import math
import numpy as np
import pytest

import sys
import os

# Ensure the parent directory is on the path so we can import the modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projection import GeneralizedPolyconicalProjection, EARTH_RADIUS_KM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def proj():
    """Default projection with normalised radius R=1 and central meridian 0°."""
    return GeneralizedPolyconicalProjection(radius=1.0, central_meridian=0.0)


@pytest.fixture
def proj_km():
    """Projection using Earth's mean radius in kilometres."""
    return GeneralizedPolyconicalProjection(
        radius=EARTH_RADIUS_KM, central_meridian=0.0
    )


# ---------------------------------------------------------------------------
# Tests: conventional_to_generalized
# ---------------------------------------------------------------------------

class TestConventionalToGeneralized:

    def test_equator_has_zero_gen_lon(self, proj):
        """At φ = 0, Λ = λ · sin(0) = 0 for any λ."""
        for lon in [-180, -90, 0, 90, 180]:
            gen_lon, gen_lat = proj.conventional_to_generalized(lon, 0.0)
            assert gen_lon == pytest.approx(0.0, abs=1e-12), (
                f"Expected gen_lon=0 at equator, got {gen_lon} for lon={lon}"
            )

    def test_gen_lat_equals_conventional_lat(self, proj):
        """Generalized latitude must equal conventional latitude for all inputs."""
        lons = np.linspace(-180, 180, 37)
        lats = np.linspace(-90, 90, 19)
        LON, LAT = np.meshgrid(lons, lats)
        _, gen_lat = proj.conventional_to_generalized(LON.ravel(), LAT.ravel())
        np.testing.assert_allclose(gen_lat, LAT.ravel(), atol=1e-12)

    def test_gen_lon_formula_at_30N(self, proj):
        """Λ = λ · sin(30°) = λ · 0.5."""
        lon, lat = 60.0, 30.0
        gen_lon, _ = proj.conventional_to_generalized(lon, lat)
        expected = lon * math.sin(math.radians(lat))
        assert gen_lon == pytest.approx(expected, rel=1e-10)

    def test_gen_lon_formula_at_90N(self, proj):
        """Λ = λ · sin(90°) = λ · 1 = λ at the North Pole."""
        lon, lat = 45.0, 90.0
        gen_lon, _ = proj.conventional_to_generalized(lon, lat)
        expected = lon * math.sin(math.radians(lat))  # = 45.0
        assert gen_lon == pytest.approx(expected, rel=1e-10)

    def test_gen_lon_is_zero_on_central_meridian(self, proj):
        """On the central meridian (λ = 0), Λ = 0 for every latitude."""
        lats = np.linspace(-89.0, 89.0, 30)
        gen_lons, _ = proj.conventional_to_generalized(
            np.zeros_like(lats), lats
        )
        np.testing.assert_allclose(gen_lons, 0.0, atol=1e-12)

    def test_central_meridian_offset(self):
        """Custom central meridian shifts the reference longitude."""
        proj90 = GeneralizedPolyconicalProjection(radius=1.0, central_meridian=90.0)
        # At lon=90, the longitude relative to central meridian is 0, so gen_lon=0
        gen_lon, _ = proj90.conventional_to_generalized(90.0, 45.0)
        assert gen_lon == pytest.approx(0.0, abs=1e-12)

    def test_scalar_and_array_consistency(self, proj):
        """Scalar and array inputs must give the same results."""
        lons = [0.0, 60.0, -120.0]
        lats = [0.0, 45.0, -30.0]
        for lon, lat in zip(lons, lats):
            g1, g2 = proj.conventional_to_generalized(lon, lat)
            g1a, g2a = proj.conventional_to_generalized(
                np.array([lon]), np.array([lat])
            )
            assert g1 == pytest.approx(float(g1a[0]), rel=1e-12)
            assert g2 == pytest.approx(float(g2a[0]), rel=1e-12)


# ---------------------------------------------------------------------------
# Tests: generalized_to_conventional
# ---------------------------------------------------------------------------

class TestGeneralizedToConventional:

    def test_identity_roundtrip(self, proj):
        """conventional_to_generalized followed by its inverse must recover inputs."""
        rng = np.random.default_rng(0)
        lons = rng.uniform(-170.0, 170.0, 100)
        lats = rng.uniform(-89.0, 89.0, 100)

        gen_lons, gen_lats = proj.conventional_to_generalized(lons, lats)
        lons_rec, lats_rec = proj.generalized_to_conventional(gen_lons, gen_lats)

        np.testing.assert_allclose(lats_rec, lats, atol=1e-10)
        # Longitude is only recoverable away from the equator
        non_eq = np.abs(lats) > 0.5
        np.testing.assert_allclose(
            lons_rec[non_eq], lons[non_eq], atol=1e-8
        )

    def test_poles_return_lat_90(self, proj):
        """Generalized lat ±90° should map to conventional lat ±90°."""
        _, lat_n = proj.generalized_to_conventional(0.0, 90.0)
        _, lat_s = proj.generalized_to_conventional(0.0, -90.0)
        assert lat_n == pytest.approx(90.0, abs=1e-12)
        assert lat_s == pytest.approx(-90.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Tests: project (forward)
# ---------------------------------------------------------------------------

class TestProject:

    def test_equator_is_straight_line(self, proj):
        """The equator (φ = 0) must project to y = 0."""
        lons = np.linspace(-180.0, 180.0, 37)
        _, y = proj.project(lons, np.zeros(37))
        np.testing.assert_allclose(y, 0.0, atol=1e-12)

    def test_equator_x_proportional_to_lon(self, proj):
        """Along the equator, x = R·λ (plate-carrée limiting case)."""
        lons = np.linspace(-180.0, 180.0, 37)
        x, _ = proj.project(lons, np.zeros(37))
        expected = proj.radius * np.radians(lons)
        np.testing.assert_allclose(x, expected, atol=1e-12)

    def test_central_meridian_projects_to_x0(self, proj):
        """The central meridian must map to x = 0 for all latitudes."""
        lats = np.linspace(-89.0, 89.0, 30)
        x, _ = proj.project(np.zeros(30), lats)
        np.testing.assert_allclose(x, 0.0, atol=1e-12)

    def test_central_meridian_y_equidistant(self, proj):
        """Along the central meridian, y = R·φ (equip-difference condition)."""
        lats = np.linspace(-80.0, 80.0, 17)
        _, y = proj.project(np.zeros(17), lats)
        expected = proj.radius * np.radians(lats)
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_north_pole_projects_to_single_point(self, proj):
        """All longitudes at lat=90° must map to the same (x, y)."""
        lons = np.linspace(-180.0, 180.0, 37)
        x, y = proj.project(lons, np.full(37, 90.0))
        # The pole is a single point on the central meridian
        np.testing.assert_allclose(x, x[0], atol=1e-10)
        np.testing.assert_allclose(y, y[0], atol=1e-10)

    def test_south_pole_projects_to_single_point(self, proj):
        """All longitudes at lat=-90° must map to the same (x, y)."""
        lons = np.linspace(-180.0, 180.0, 37)
        x, y = proj.project(lons, np.full(37, -90.0))
        np.testing.assert_allclose(x, x[0], atol=1e-10)
        np.testing.assert_allclose(y, y[0], atol=1e-10)

    def test_symmetry_north_south(self, proj):
        """The projection must be antisymmetric about the equator."""
        lons = np.array([0.0, 45.0, 90.0, -120.0])
        lats = np.array([30.0, 45.0, 60.0, 15.0])
        x_n, y_n = proj.project(lons, lats)
        x_s, y_s = proj.project(lons, -lats)
        np.testing.assert_allclose(x_n, x_s, atol=1e-12)
        np.testing.assert_allclose(y_n, -y_s, atol=1e-12)

    def test_symmetry_east_west(self, proj):
        """The projection must be antisymmetric about the central meridian."""
        lats = np.array([30.0, 45.0, 60.0, 15.0])
        lons = np.array([60.0, 45.0, 30.0, 90.0])
        x_e, y_e = proj.project(lons, lats)
        x_w, y_w = proj.project(-lons, lats)
        np.testing.assert_allclose(x_e, -x_w, atol=1e-12)
        np.testing.assert_allclose(y_e, y_w, atol=1e-12)

    def test_scalar_inputs(self, proj):
        """Scalar inputs should return floats, not arrays."""
        x, y = proj.project(0.0, 45.0)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_formula_at_30N_60E(self, proj):
        """Verify the polyconical formula numerically at (60°E, 30°N)."""
        lon, lat = 60.0, 30.0
        x, y = proj.project(lon, lat)

        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        rho = proj.radius / math.tan(lat_r)
        gen_lon_r = lon_r * math.sin(lat_r)
        x_expected = rho * math.sin(gen_lon_r)
        y_expected = proj.radius * lat_r - rho * (1.0 - math.cos(gen_lon_r))

        assert x == pytest.approx(x_expected, rel=1e-10)
        assert y == pytest.approx(y_expected, rel=1e-10)

    def test_km_scale(self, proj_km):
        """Projected coordinates with R = 6371 km should scale linearly."""
        x_km, y_km = proj_km.project(0.0, 45.0)
        proj_unit = GeneralizedPolyconicalProjection(radius=1.0)
        x_1, y_1 = proj_unit.project(0.0, 45.0)
        assert x_km == pytest.approx(x_1 * EARTH_RADIUS_KM, rel=1e-10)
        assert y_km == pytest.approx(y_1 * EARTH_RADIUS_KM, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests: inverse_project (round-trip)
# ---------------------------------------------------------------------------

class TestInverseProject:

    def test_central_meridian_exact(self, proj):
        """On the central meridian the inverse is exact."""
        lats = np.linspace(-80.0, 80.0, 17)
        x = np.zeros_like(lats)
        y = proj.radius * np.radians(lats)
        lon_rec, lat_rec = proj.inverse_project(x, y)
        np.testing.assert_allclose(lat_rec, lats, atol=1e-8)
        np.testing.assert_allclose(lon_rec, 0.0, atol=1e-8)

    def test_random_round_trip(self, proj):
        """The inverse must satisfy forward(inverse(x, y)) ≈ (x, y).

        The polyconic projection is not globally injective: different (lon, lat)
        pairs can project to the same (x, y), so inverse(forward(lon, lat))
        need not return the original (lon, lat).  The correct invariant is
        that re-projecting the recovered coordinates must reproduce the
        original projected coordinates.
        """
        rng = np.random.default_rng(7)
        lons = rng.uniform(-90.0, 90.0, 300)
        lats = rng.uniform(-80.0, 80.0, 300)

        x, y = proj.project(lons, lats)
        lons_rec, lats_rec = proj.inverse_project(x, y)
        x_back, y_back = proj.project(lons_rec, lats_rec)

        np.testing.assert_allclose(x_back, x, atol=1e-6)
        np.testing.assert_allclose(y_back, y, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: projection properties / mathematical invariants
# ---------------------------------------------------------------------------

class TestProjectionProperties:

    def test_parallels_increase_with_latitude(self, proj):
        """Higher latitudes should have higher y-values on the central meridian."""
        lats = np.arange(-80.0, 81.0, 10.0)
        _, y = proj.project(np.zeros_like(lats), lats)
        diffs = np.diff(y)
        assert np.all(diffs > 0), "Parallels are not monotonically increasing"

    def test_equip_difference_spacing(self, proj):
        """Spacing between successive parallels along central meridian must be equal."""
        lats = np.arange(-80.0, 81.0, 10.0)
        _, y = proj.project(np.zeros_like(lats), lats)
        spacing = np.diff(y)
        expected = proj.radius * math.radians(10.0)
        np.testing.assert_allclose(spacing, expected, rtol=1e-8)

    def test_meridian_lengths_scale_with_radius(self):
        """Pole-to-equator distance on central meridian must equal R · π/2."""
        for R in [1.0, 6371.0, 100.0]:
            p = GeneralizedPolyconicalProjection(radius=R)
            _, y_pole = p.project(0.0, 90.0)
            expected = R * math.pi / 2.0
            assert y_pole == pytest.approx(expected, rel=1e-10)

    def test_earth_radius_constant(self):
        """The default Earth radius constant must be 6371 km."""
        assert EARTH_RADIUS_KM == pytest.approx(6371.0)
