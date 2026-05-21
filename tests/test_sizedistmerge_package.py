import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_import_sizedistmerge_from_outside_repo(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    code = """
import sys
import sizedistmerge as sdm
assert "miepython" not in sys.modules
assert sdm.edges_from_mids_geometric([10, 20, 40]).shape == (4,)
assert isinstance(sdm.da_to_dv(100.0, rho_p=1000.0), float)
assert callable(sdm.calculate_dry_diameter)
assert callable(sdm.kappa_from_growth_factor)
assert "miepython" not in sys.modules
assert callable(sdm.POPSGeom)
assert "miepython" in sys.modules
assert callable(sdm.merge_sizedists_tikhonov_consensus)
assert callable(sdm.temporal_parameter_penalty)
assert callable(sdm.objective_joint_named_temporal)
assert callable(sdm.number_to_surface_area_spectrum)
assert not hasattr(sdm, "twomey_inversion")
assert callable(sdm.arcsix_merge_production.run_arcsix_merge_for_periods)
assert callable(sdm.arcsix_merge_production.run_post_merge_product_qc)
assert not hasattr(sdm, "merge_production")
assert not hasattr(sdm, "arcsix_product")
assert sdm.lut_path("uhsas").name == "uhsas_sigma_col_1054nm.zarr"
assert sdm.lut_path("pops").name == "pops_sigma_col_405nm.zarr"
assert "sizedistmerge/data/lut" in sdm.lut_path("uhsas").as_posix()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_count_conserving_rebin_preserves_total_counts():
    sys.path.insert(0, str(SRC))
    import sizedistmerge as sdm

    old_edges = np.array([10.0, 100.0, 1000.0])
    new_edges = np.array([10.0, 31.6227766017, 100.0, 316.227766017, 1000.0])
    old_y = np.array([5.0, 10.0])

    new_y = sdm.rebin_dndlog_by_edges_overlap(old_edges, new_edges, old_y)

    old_counts = np.sum(old_y * np.diff(np.log10(old_edges)))
    new_counts = np.nansum(new_y * np.diff(np.log10(new_edges)))
    assert np.isclose(new_counts, old_counts)


def test_validation_errors_are_clear_for_bad_grids():
    sys.path.insert(0, str(SRC))
    import sizedistmerge as sdm

    try:
        sdm.edges_from_mids_geometric([20.0, 10.0])
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-increasing mids")


def test_package_has_no_twomey_module():
    assert not (SRC / "sizedistmerge" / "twomey.py").exists()
    assert not (SRC / "sizedistmerge" / "twomey_inversion.py").exists()


def test_arcsix_merge_production_has_single_canonical_api():
    sys.path.insert(0, str(SRC))
    from sizedistmerge import arcsix_merge_production as mp

    assert "as_timezone_naive_timestamp" in mp.__all__
    assert "drop_timezone_from_index" in mp.__all__
    assert "load_arcsix_merge_frames_for_day" in mp.__all__
    assert "periods_from_frames" in mp.__all__
    assert "run_post_merge_product_qc" in mp.__all__
    assert "convert_qc_netcdf_to_icartt" in mp.__all__
    assert "run_joint_optimization" in mp.__all__
    assert "make_consensus_merged_spec" in mp.__all__
    assert "write_day_netcdf" in mp.__all__
    assert "run_arcsix_merge_for_periods" in mp.__all__
    assert "run_arcsix_merge_for_periods_fims_uhsas_aps" not in mp.__all__
    assert "run_arcsix_merge_for_periods_fims_aps_only" not in mp.__all__
    assert not any("v2" in name.lower() for name in mp.__all__)
    assert not hasattr(mp, "run_joint_optimization_v2")
    assert not hasattr(mp, "write_day_netcdf_v2")
    assert not hasattr(mp, "run_arcsix_merge_for_periods_fims_uhsas_aps")
    assert not hasattr(mp, "run_arcsix_merge_for_periods_fims_aps_only")

    source = (SRC / "sizedistmerge" / "arcsix_merge_production.py").read_text()
    assert "_v2" not in source
    assert "BASE_DIR" not in source


def test_arcsix_production_has_no_compatibility_modules():
    assert not (SRC / "sizedistmerge" / "merge_production.py").exists()
    assert not (SRC / "sizedistmerge" / "arcsix_product.py").exists()


def test_arcsix_time_helpers_strip_timezone_without_clock_shift():
    sys.path.insert(0, str(SRC))
    import pandas as pd
    from sizedistmerge import arcsix_merge_production as mp

    ts = mp.as_timezone_naive_timestamp("2024-08-02 11:09:37-06:00")
    assert ts == pd.Timestamp("2024-08-02 11:09:37")

    idx = pd.DatetimeIndex(["2024-08-02 11:09:37-06:00", "2024-08-02 11:09:38-06:00"])
    df = pd.DataFrame({"x": [1, 2]}, index=idx)
    out = mp.drop_timezone_from_index(df)

    assert out.index.tz is None
    assert out.index.tolist() == [
        pd.Timestamp("2024-08-02 11:09:37"),
        pd.Timestamp("2024-08-02 11:09:38"),
    ]


def test_arcsix_period_helpers_reuse_existing_split_behavior():
    sys.path.insert(0, str(SRC))
    import pandas as pd
    from sizedistmerge import arcsix_merge_production as mp

    fims = pd.DataFrame(
        {"x": [1, 2, 3]},
        index=pd.to_datetime(
            [
                "2024-08-02 11:00:00-06:00",
                "2024-08-02 11:00:30-06:00",
                "2024-08-02 11:01:00-06:00",
            ]
        ),
    )
    aps = pd.DataFrame(
        {"x": [4, 5]},
        index=pd.to_datetime(
            [
                "2024-08-02 11:00:10-06:00",
                "2024-08-02 11:01:10-06:00",
            ]
        ),
    )

    periods = mp.periods_from_frames({"FIMS": fims, "APS": aps}, 60)

    assert periods == [
        (pd.Timestamp("2024-08-02 11:00:00"), pd.Timestamp("2024-08-02 11:00:30")),
        (pd.Timestamp("2024-08-02 11:01:00"), pd.Timestamp("2024-08-02 11:01:10")),
    ]


def test_load_arcsix_merge_frames_for_day_applies_dirs_timezone_and_fims_lag(monkeypatch, tmp_path):
    sys.path.insert(0, str(SRC))
    import pandas as pd
    from sizedistmerge import arcsix_merge_production as mp

    captured = {}
    idx = pd.DatetimeIndex(["2024-08-02 11:00:10-06:00", "2024-08-02 11:00:20-06:00"])

    def fake_load(date, aps_dir, uhsas_dir, fims_dir, pops_dir=None):
        captured["date"] = date
        captured["aps_dir"] = aps_dir
        captured["uhsas_dir"] = uhsas_dir
        captured["fims_dir"] = fims_dir
        captured["pops_dir"] = pops_dir
        return {
            "FIMS": pd.DataFrame({"x": [1, 2]}, index=idx),
            "APS": pd.DataFrame({"x": [3, 4]}, index=idx),
        }

    monkeypatch.setattr(mp, "load_aufi_oneday", fake_load)

    frames = mp.load_arcsix_merge_frames_for_day(
        tmp_path,
        "2024-08-02",
        instruments=("FIMS", "APS"),
        fims_lag=10,
    )

    assert captured["date"] == "2024-08-02"
    assert captured["aps_dir"] == tmp_path / "LARGE-APS"
    assert captured["uhsas_dir"] is None
    assert captured["fims_dir"] == tmp_path / "FIMS"
    assert captured["pops_dir"] is None
    assert frames["FIMS"].index.tz is None
    assert frames["APS"].index.tz is None
    assert frames["FIMS"].index.tolist() == [
        pd.Timestamp("2024-08-02 11:00:00"),
        pd.Timestamp("2024-08-02 11:00:10"),
    ]
    assert frames["APS"].index.tolist() == [
        pd.Timestamp("2024-08-02 11:00:10"),
        pd.Timestamp("2024-08-02 11:00:20"),
    ]


def test_ict_time_helpers_strip_timezone_without_clock_shift():
    sys.path.insert(0, str(SRC))
    import pandas as pd
    from sizedistmerge import ict_utils as ict

    ts = ict.parse_bound("2024-08-02 11:09:37-06:00")
    assert ts == pd.Timestamp("2024-08-02 11:09:37")

    day0, day1 = ict._local_day_bounds("2024-08-02 11:09:37-06:00")
    assert day0 == pd.Timestamp("2024-08-02 00:00:00")
    assert day1 == pd.Timestamp("2024-08-03 00:00:00")

    df = pd.DataFrame({"Time_Start": [40177.0]})
    out = ict._attach_time(df, meas_date=pd.Timestamp("2024-08-02").to_pydatetime())

    assert out["time"].dt.tz is None
    assert out.loc[0, "time"] == pd.Timestamp("2024-08-02 11:09:37")

    frame = pd.DataFrame(
        {"value": [1.0]},
        index=pd.DatetimeIndex(["2024-08-02 11:09:37-06:00"]),
    )
    aligned, grid = ict.align_to_common_grid({"x": frame}, mode="union", round_to=None)
    assert grid.tz is None
    assert grid[0] == pd.Timestamp("2024-08-02 11:09:37")
    assert aligned["x"].index.tz is None


def test_temporal_penalty_and_no_overlap_are_explicit():
    sys.path.insert(0, str(SRC))
    from sizedistmerge import alignment as al

    penalty = al.temporal_parameter_penalty(
        np.array([1.50, 1200.0]),
        np.array([1.40, 1000.0]),
        np.array([10.0, 1e-7]),
    )
    assert np.isclose(penalty, 0.104)

    def identity_edges(edges, _theta, **_kwargs):
        return np.asarray(edges, float)

    cost = al.objective_multi_custom(
        np.array([1.0]),
        np.array([10.0, 20.0]),
        np.array([1.0, 1.0]),
        [
            {
                "edges": np.array([1000.0, 2000.0, 3000.0]),
                "y": np.array([1.0, 1.0]),
                "remap_fn": identity_edges,
                "kwargs": {},
                "w_ref": 1.0,
            }
        ],
        [slice(0, 1)],
    )
    assert cost == 0.0


def test_mse_overlap_keeps_zero_bins_for_legacy_parity():
    sys.path.insert(0, str(SRC))
    from sizedistmerge import alignment as al

    x = np.array([10.0, 20.0, 40.0])
    y_with_zero = np.array([5.0, 0.0, 5.0])
    y_other = np.array([5.0, 10.0, 5.0])

    cost = al.mse_overlap_sizedist(
        x,
        y_with_zero,
        x,
        y_other,
        moment="N",
        space="linear",
    )

    log_grid = np.linspace(np.log10(10.0), np.log10(40.0), 128)
    expected = np.mean(
        (
            np.interp(log_grid, np.log10(x), y_with_zero)
            - np.interp(log_grid, np.log10(x), y_other)
        )
        ** 2
    )
    dropped_zero_expected = np.mean(
        (
            np.interp(log_grid, np.log10([10.0, 40.0]), [5.0, 5.0])
            - np.interp(log_grid, np.log10(x), y_other)
        )
        ** 2
    )

    assert np.isclose(cost, expected)
    assert not np.isclose(cost, dropped_zero_expected)


def test_named_temporal_objective_matches_notebook_shape():
    sys.path.insert(0, str(SRC))
    from sizedistmerge import alignment as al

    edges = np.array([10.0, 20.0, 40.0])
    y = np.array([1.0, 2.0])
    mids = np.array([14.14213562, 28.28427125])

    def identity_edges(edges_in, _theta, **_kwargs):
        return np.asarray(edges_in, float)

    cost = al.objective_joint_named_temporal(
        np.array([1.50, 1.60, 1200.0]),
        ref_mids=mids,
        ref_y=y,
        uhsas_edges=edges,
        uhsas_y=y,
        pops_edges=edges,
        pops_y=y,
        aps_edges=edges,
        aps_y=y,
        uhsas_remap_fn=identity_edges,
        uhsas_kwargs={},
        pops_remap_fn=identity_edges,
        pops_kwargs={},
        aps_remap_fn=identity_edges,
        aps_kwargs={},
        prev_params=np.array([1.40, 1.50, 1000.0]),
        temporal_w_uh=10.0,
        temporal_w_po=10.0,
        temporal_w_rho=1e-7,
        pair_w=1.0,
    )
    assert np.isclose(cost, 0.204)

    no_overlap_cost = al.objective_joint_named_temporal(
        np.array([1.50, 1.60, 1200.0]),
        ref_mids=np.array([1.0, 2.0]),
        ref_y=y,
        uhsas_edges=np.array([1000.0, 2000.0, 3000.0]),
        uhsas_y=y,
        pops_edges=np.array([1000.0, 2000.0, 3000.0]),
        pops_y=y,
        aps_edges=np.array([1000.0, 2000.0, 3000.0]),
        aps_y=y,
        uhsas_remap_fn=identity_edges,
        uhsas_kwargs={},
        pops_remap_fn=identity_edges,
        pops_kwargs={},
        aps_remap_fn=identity_edges,
        aps_kwargs={},
        prev_params=np.array([1.40, 1.50, 1000.0]),
        temporal_w_uh=10.0,
        temporal_w_po=10.0,
        temporal_w_rho=1e-7,
        pair_w=0.0,
    )
    assert no_overlap_cost == 0.0


def test_optimize_multi_custom_tracks_temporal_cost(monkeypatch):
    sys.path.insert(0, str(SRC))
    from sizedistmerge import alignment as al

    def identity_edges(edges, _theta, **_kwargs):
        return np.asarray(edges, float)

    class Result:
        x = np.array([1.50, 1200.0])
        fun = 0.104

    def fake_de(obj, bounds, callback=None, **_kwargs):
        x = np.array([1.50, 1200.0])
        if callback is not None:
            callback(x, 0.0)
        value = obj(x)
        assert np.isclose(value, 0.104)
        assert bounds == [(1.3, 1.8), (950.0, 2000.0)]
        return Result()

    monkeypatch.setattr(al, "differential_evolution", fake_de)

    best_thetas, best_cost, _res, hist = al.optimize_multi_custom(
        ref_mids=np.array([14.14213562, 28.28427125]),
        ref_y=np.array([1.0, 2.0]),
        instruments=[
            {
                "edges": np.array([10.0, 20.0, 40.0]),
                "y": np.array([1.0, 2.0]),
                "remap_fn": identity_edges,
                "kwargs": {},
                "w_ref": 1.0,
            },
            {
                "edges": np.array([10.0, 20.0, 40.0]),
                "y": np.array([1.0, 2.0]),
                "remap_fn": identity_edges,
                "kwargs": {},
                "w_ref": 1.0,
            },
        ],
        bounds_list=[[(1.3, 1.8)], [(950.0, 2000.0)]],
        temporal_target=np.array([1.40, 1000.0]),
        temporal_weights=np.array([10.0, 1e-7]),
    )

    assert np.isclose(best_cost, 0.104)
    assert np.isclose(hist["best_data_cost"], 0.0)
    assert np.isclose(hist["best_temporal_cost"], 0.104)
    assert np.isclose(hist["total"][0], 0.104)
    assert [float(theta[0]) for theta in best_thetas] == [1.50, 1200.0]


def test_optimize_multi_custom_skips_temporal_when_nothing_overlaps(monkeypatch):
    sys.path.insert(0, str(SRC))
    from sizedistmerge import alignment as al

    def identity_edges(edges, _theta, **_kwargs):
        return np.asarray(edges, float)

    class Result:
        x = np.array([1.50])
        fun = 0.0

    def fake_de(obj, bounds, callback=None, **_kwargs):
        x = np.array([1.50])
        if callback is not None:
            callback(x, 0.0)
        assert obj(x) == 0.0
        assert bounds == [(1.3, 1.8)]
        return Result()

    monkeypatch.setattr(al, "differential_evolution", fake_de)

    _best_thetas, best_cost, _res, hist = al.optimize_multi_custom(
        ref_mids=np.array([1.0, 2.0]),
        ref_y=np.array([1.0, 2.0]),
        instruments=[
            {
                "edges": np.array([1000.0, 2000.0, 3000.0]),
                "y": np.array([1.0, 2.0]),
                "remap_fn": identity_edges,
                "kwargs": {},
                "w_ref": 1.0,
            },
        ],
        bounds_list=[[(1.3, 1.8)]],
        temporal_target=np.array([1.40]),
        temporal_weights=np.array([10.0]),
    )

    assert best_cost == 0.0
    assert hist["best_data_cost"] == 0.0
    assert hist["best_temporal_cost"] == 0.0
    assert hist["total"] == [0.0]


def test_run_joint_optimization_uses_single_multi_instrument_path(monkeypatch):
    sys.path.insert(0, str(SRC))
    from sizedistmerge import arcsix_merge_production as mp

    edges = np.array([10.0, 20.0, 40.0])
    mids = np.array([14.14213562, 28.28427125])
    y = np.array([1.0, 2.0])
    sigma = np.array([0.1, 0.2])
    specs = {
        "FIMS": (mids, edges, y, sigma),
        "UHSAS": (mids, edges, y, sigma),
        "POPS": (mids, edges, y, sigma),
        "APS": (mids, edges, y, sigma),
    }
    line_kwargs = {}
    fill_kwargs = {}
    captured = {}

    monkeypatch.setattr(mp, "_load_uhsas_lut", lambda _lut_dir: object())
    monkeypatch.setattr(mp, "_load_pops_lut", lambda _lut_dir: object())

    def fake_remap(edges_in, theta, **_kwargs):
        return np.asarray(edges_in, float) * (1.0 + float(theta[0]) * 1e-6)

    monkeypatch.setattr(mp, "_uhsas_remap_fn", fake_remap)
    monkeypatch.setattr(mp, "_pops_remap_fn", fake_remap)
    monkeypatch.setattr(mp, "_aps_remap_fn", fake_remap)

    def fake_optimize_multi_custom(**kwargs):
        captured["instrument_count"] = len(kwargs["instruments"])
        captured["pair_weights"] = kwargs["pair_weights"]
        captured["w_refs"] = [inst["w_ref"] for inst in kwargs["instruments"]]
        captured["temporal_target"] = kwargs["temporal_target"]
        captured["temporal_weights"] = kwargs["temporal_weights"]
        captured["ri_srcs"] = [
            inst["kwargs"].get("ri_src")
            for inst in kwargs["instruments"]
            if "ri_src" in inst["kwargs"]
        ]
        return (
            [np.array([1.45]), np.array([1.55]), np.array([1200.0])],
            0.25,
            object(),
            {"total": [1.0, 0.25], "best_data_cost": 0.20, "best_temporal_cost": 0.05},
        )

    monkeypatch.setattr(mp, "optimize_multi_custom", fake_optimize_multi_custom)

    specs_out, _, _, opt_res = mp.run_joint_optimization(
        specs,
        line_kwargs,
        fill_kwargs,
        uhsas_xmin=None,
        w_uhsas=2.0,
        w_pops=3.0,
        w_aps=4.0,
        prev_params=[1.40, 1.60, 1000.0],
        temporal_w_uh=10.0,
        temporal_w_po=11.0,
        temporal_w_rho=1e-7,
    )

    assert captured["instrument_count"] == 3
    assert captured["pair_weights"] == [(0, 2, 1.0), (1, 2, 1.0)]
    assert captured["w_refs"] == [2.0, 3.0, 4.0]
    assert captured["temporal_target"].tolist() == [1.40, 1.60, 1000.0]
    assert captured["temporal_weights"].tolist() == [10.0, 11.0, 1e-7]
    assert captured["ri_srcs"] == [mp.RI_UHSAS_SRC, mp.RI_UHSAS_SRC]
    assert opt_res["n_fit"] == 1.45
    assert opt_res["n_pops_fit"] == 1.55
    assert opt_res["rho_fit"] == 1200.0
    assert opt_res["data_cost"] == 0.20
    assert opt_res["temporal_cost"] == 0.05
    assert set(opt_res["fit_labels"]) == {"UHSAS", "POPS", "APS"}
    for label in opt_res["fit_labels"].values():
        assert label in specs_out


def test_run_joint_optimization_handles_fims_uhsas_aps_without_pops(monkeypatch):
    sys.path.insert(0, str(SRC))
    from sizedistmerge import arcsix_merge_production as mp

    edges = np.array([10.0, 20.0, 40.0])
    mids = np.array([14.14213562, 28.28427125])
    y = np.array([1.0, 2.0])
    sigma = np.array([0.1, 0.2])
    specs = {
        "FIMS": (mids, edges, y, sigma),
        "UHSAS": (mids, edges, y, sigma),
        "APS": (mids, edges, y, sigma),
    }
    captured = {}

    monkeypatch.setattr(mp, "_load_uhsas_lut", lambda _lut_dir: object())

    def fail_pops_lut(_lut_dir):
        raise AssertionError("POPS LUT should not load in FIMS+UHSAS+APS mode")

    monkeypatch.setattr(mp, "_load_pops_lut", fail_pops_lut)

    def fake_remap(edges_in, theta, **_kwargs):
        return np.asarray(edges_in, float) * (1.0 + float(theta[0]) * 1e-6)

    monkeypatch.setattr(mp, "_uhsas_remap_fn", fake_remap)
    monkeypatch.setattr(mp, "_aps_remap_fn", fake_remap)

    def fake_optimize_multi_custom(**kwargs):
        captured["instrument_count"] = len(kwargs["instruments"])
        captured["pair_weights"] = kwargs["pair_weights"]
        captured["temporal_target"] = kwargs["temporal_target"]
        captured["temporal_weights"] = kwargs["temporal_weights"]
        captured["ri_srcs"] = [
            inst["kwargs"].get("ri_src")
            for inst in kwargs["instruments"]
            if "ri_src" in inst["kwargs"]
        ]
        return (
            [np.array([1.45]), np.array([1200.0])],
            0.25,
            object(),
            {"total": [1.0, 0.25]},
        )

    monkeypatch.setattr(mp, "optimize_multi_custom", fake_optimize_multi_custom)

    specs_out, _, _, opt_res = mp.run_joint_optimization(
        specs,
        {},
        {},
        uhsas_xmin=None,
        prev_params=[1.40, np.nan, 1000.0],
        temporal_w_uh=10.0,
        temporal_w_po=11.0,
        temporal_w_rho=1e-7,
    )

    assert captured["instrument_count"] == 2
    assert captured["pair_weights"] == [(0, 1, 1.0)]
    assert captured["temporal_target"].tolist() == [1.40, 1000.0]
    assert captured["temporal_weights"].tolist() == [10.0, 1e-7]
    assert captured["ri_srcs"] == [mp.RI_UHSAS_SRC]
    assert opt_res["n_fit"] == 1.45
    assert np.isnan(opt_res["n_pops_fit"])
    assert opt_res["rho_fit"] == 1200.0
    assert set(opt_res["fit_labels"]) == {"UHSAS", "APS"}
    for label in opt_res["fit_labels"].values():
        assert label in specs_out


def test_run_joint_optimization_can_override_pops_ri_source(monkeypatch):
    sys.path.insert(0, str(SRC))
    from sizedistmerge import arcsix_merge_production as mp

    edges = np.array([10.0, 20.0, 40.0])
    mids = np.array([14.14213562, 28.28427125])
    y = np.array([1.0, 2.0])
    sigma = np.array([0.1, 0.2])
    specs = {
        "FIMS": (mids, edges, y, sigma),
        "POPS": (mids, edges, y, sigma),
    }
    captured = {}

    monkeypatch.setattr(mp, "_load_pops_lut", lambda _lut_dir: object())

    def fake_remap(edges_in, theta, **kwargs):
        captured.setdefault("ri_srcs", []).append(kwargs["ri_src"])
        return np.asarray(edges_in, float) * (1.0 + float(theta[0]) * 1e-6)

    monkeypatch.setattr(mp, "_pops_remap_fn", fake_remap)

    def fake_optimize_multi_custom(**kwargs):
        captured["opt_ri_src"] = kwargs["instruments"][0]["kwargs"]["ri_src"]
        return (
            [np.array([1.55])],
            0.25,
            object(),
            {"total": [0.25]},
        )

    monkeypatch.setattr(mp, "optimize_multi_custom", fake_optimize_multi_custom)

    _specs_out, _line, _fill, opt_res = mp.run_joint_optimization(
        specs,
        {},
        {},
        pops_ri_src=mp.RI_POPS_SRC,
        fims_xmax=500,
    )

    assert captured["opt_ri_src"] == mp.RI_POPS_SRC
    assert captured["ri_srcs"] == [mp.RI_POPS_SRC]
    assert opt_res["pops_ri_src"] == mp.RI_POPS_SRC


def test_arcsix_instrument_selection_is_configurable():
    sys.path.insert(0, str(SRC))
    from sizedistmerge import arcsix_merge_production as mp

    assert mp._normalize_merge_instruments(None, include_pops=True) == ("FIMS", "UHSAS", "POPS", "APS")
    assert mp._normalize_merge_instruments(None, include_pops=False) == ("FIMS", "UHSAS", "APS")
    assert mp._normalize_merge_instruments(("FIMS", "APS"), include_pops=True) == ("FIMS", "APS")
    assert mp._normalize_merge_instruments(("FIMS", "UHSAS", "APS"), include_pops=True) == ("FIMS", "UHSAS", "APS")
    assert mp._validate_apply_alignment(True, ("FIMS", "APS")) is True
    assert mp._validate_apply_alignment(False, ("FIMS", "APS")) is False

    try:
        mp._normalize_merge_instruments(("FIMS", "POPS"), include_pops=True)
    except ValueError as exc:
        assert "supported ARCSIX instrument sets" in str(exc)
    else:
        raise AssertionError("expected unsupported instrument set to fail")

    try:
        mp._validate_apply_alignment(False, ("FIMS", "UHSAS", "APS"))
    except ValueError as exc:
        assert "apply_alignment=False" in str(exc)
    else:
        raise AssertionError("expected no-alignment non-FIMS+APS mode to fail")


def test_arcsix_period_runner_carries_temporal_params(monkeypatch, tmp_path):
    sys.path.insert(0, str(SRC))
    import matplotlib.pyplot as plt
    import pandas as pd
    from sizedistmerge import arcsix_merge_production as mp

    idx = pd.date_range("2024-05-28 00:00:00.500", periods=4, freq="30s")
    frames = {
        name: pd.DataFrame({"placeholder": np.ones(len(idx))}, index=idx)
        for name in ("APS", "UHSAS", "POPS", "FIMS")
    }
    edges = np.array([10.0, 20.0, 40.0])
    mids = np.array([14.14213562, 28.28427125])
    y = np.array([1.0, 2.0])
    sigma = np.array([0.1, 0.2])
    base_specs = {
        "FIMS": (mids, edges, y, sigma),
        "UHSAS": (mids, edges, y, sigma),
        "POPS": (mids, edges, y, sigma),
        "APS": (mids, edges, y, sigma),
    }
    captured_prev = []
    captured_joint_kwargs = []
    captured_consensus_kwargs = {}
    written = {}

    monkeypatch.setattr(mp, "load_aufi_oneday", lambda *_args, **_kwargs: {k: v.copy() for k, v in frames.items()})
    monkeypatch.setattr(
        mp,
        "read_inlet_flag",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"InletFlag_LARGE": np.zeros(120, dtype=int)},
            index=pd.date_range("2024-05-28 00:00:00", periods=120, freq="s"),
        ),
    )
    monkeypatch.setattr(
        mp,
        "read_microphysical",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"CNgt10nm": np.ones(120)},
            index=pd.date_range("2024-05-28 00:00:00", periods=120, freq="s"),
        ),
    )
    monkeypatch.setattr(mp, "plot_period_totals", lambda *_args, **_kwargs: plt.subplots())
    monkeypatch.setattr(mp, "chunk_is_incloud", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        mp,
        "make_filtered_specs",
        lambda *_args, **_kwargs: (
            dict(base_specs),
            {},
            {},
            {name: np.array([10, 10]) for name in base_specs},
        ),
    )

    def fake_run_joint_optimization(specs, line_kwargs, fill_kwargs, **kwargs):
        captured_joint_kwargs.append(kwargs)
        captured_prev.append(np.asarray(kwargs["prev_params"], dtype=float).copy())
        call_idx = len(captured_prev)
        n_uh = 1.44 + 0.01 * call_idx
        n_po = 1.54 + 0.01 * call_idx
        rho = 1190.0 + 10.0 * call_idx
        specs_out = dict(specs)
        specs_out["FIMS_applied"] = specs["FIMS"]
        uh_label = f"UHSAS fit (n={n_uh:.3f})"
        po_label = f"POPS fit (n={n_po:.3f})"
        aps_label = f"APS fit (ρ={rho*0.001:.3f} g/cm$^3$)"
        specs_out[uh_label] = specs["UHSAS"]
        specs_out[po_label] = specs["POPS"]
        specs_out[aps_label] = specs["APS"]
        return (
            specs_out,
            dict(line_kwargs),
            dict(fill_kwargs),
            {
                "n_fit": n_uh,
                "n_pops_fit": n_po,
                "rho_fit": rho,
                "best_cost": 0.2,
                "data_cost": 0.15,
                "temporal_cost": 0.05,
                "hist": {"total": [0.2], "data": [0.15], "temporal": [0.05]},
                "fit_labels": {"UHSAS": uh_label, "POPS": po_label, "APS": aps_label},
            },
        )

    monkeypatch.setattr(mp, "run_joint_optimization", fake_run_joint_optimization)
    monkeypatch.setattr(mp, "plot_history", lambda *_args, **_kwargs: plt.subplots())
    def fake_make_consensus_merged_spec(**kwargs):
        captured_consensus_kwargs.update(kwargs)
        return (
            {"merged": (mids, edges, y, sigma)},
            {},
            {},
            {},
        )

    monkeypatch.setattr(mp, "make_consensus_merged_spec", fake_make_consensus_merged_spec)
    monkeypatch.setattr(mp, "plot_sizedist_all", lambda **_kwargs: (plt.subplots(), plt.subplots(), ({}, {})))
    monkeypatch.setattr(mp, "remap_dndlog_by_edges_any", lambda _old, _new, vals: np.asarray(vals, float))

    def fake_write_day_netcdf(*_args, **kwargs):
        written.update(kwargs)
        return tmp_path / "out.nc"

    monkeypatch.setattr(mp, "write_day_netcdf", fake_write_day_netcdf)

    mp.run_arcsix_merge_for_periods(
        [
            ("2024-05-28 00:00:00", "2024-05-28 00:00:59"),
            ("2024-05-28 00:01:00", "2024-05-28 00:01:59"),
        ],
        tmp_path / "data",
        tmp_path / "out",
        fims_lag=0,
        min_samples_per_inst=1,
        min_overlap_s=1,
        overlap_freq="1s",
        space="log",
        consensus_data_space="log10",
        temporal_prior_params=[1.52, 1.615, 1000.0],
        temporal_w_uh=10.0,
        temporal_w_po=10.0,
        temporal_w_rho=1e-7,
        aps_combine_weight=2.0,
        output_edges=edges,
    )

    assert len(captured_prev) == 2
    assert captured_joint_kwargs[0]["space"] == "log"
    assert captured_consensus_kwargs["data_space"] == "log10"
    assert captured_consensus_kwargs["alpha_aps"] == 2.0
    assert captured_prev[0].tolist() == [1.52, 1.615, 1000.0]
    assert captured_prev[1].tolist() == [1.45, 1.55, 1200.0]
    assert written["day_fine_edges"].tolist() == edges.tolist()
    assert written["day_n_fit"].tolist() == [1.45, 1.46]
    assert written["day_n_pops_fit"].tolist() == [1.55, 1.56]
    assert written["day_rho_fit"].tolist() == [1200.0, 1210.0]


def test_temporal_regularization_requires_explicit_prior(monkeypatch):
    sys.path.insert(0, str(SRC))
    from sizedistmerge import arcsix_merge_production as mp

    edges = np.array([10.0, 20.0, 40.0])
    mids = np.array([14.14213562, 28.28427125])
    y = np.array([1.0, 2.0])
    sigma = np.array([0.1, 0.2])
    specs = {
        "FIMS": (mids, edges, y, sigma),
        "UHSAS": (mids, edges, y, sigma),
        "APS": (mids, edges, y, sigma),
    }

    monkeypatch.setattr(mp, "_load_uhsas_lut", lambda _lut_dir: object())
    monkeypatch.setattr(mp, "_uhsas_remap_fn", lambda edges_in, _theta, **_kwargs: np.asarray(edges_in, float))
    monkeypatch.setattr(mp, "_aps_remap_fn", lambda edges_in, _theta, **_kwargs: np.asarray(edges_in, float))

    try:
        mp.run_joint_optimization(
            specs,
            {},
            {},
            uhsas_xmin=None,
            temporal_w_uh=1.0,
        )
    except ValueError as exc:
        assert "requires explicit" in str(exc)
    else:
        raise AssertionError("expected temporal regularization without prev_params to fail")


def test_arcsix_period_runner_raises_after_chunk_errors(monkeypatch, tmp_path):
    sys.path.insert(0, str(SRC))
    import matplotlib.pyplot as plt
    import pandas as pd
    from sizedistmerge import arcsix_merge_production as mp

    idx = pd.date_range("2024-05-28 00:00:00.500", periods=2, freq="30s")
    frames = {
        name: pd.DataFrame({"placeholder": np.ones(len(idx))}, index=idx)
        for name in ("APS", "UHSAS", "POPS", "FIMS")
    }
    edges = np.array([10.0, 20.0, 40.0])
    mids = np.array([14.14213562, 28.28427125])
    y = np.array([1.0, 2.0])
    sigma = np.array([0.1, 0.2])
    base_specs = {
        "FIMS": (mids, edges, y, sigma),
        "UHSAS": (mids, edges, y, sigma),
        "POPS": (mids, edges, y, sigma),
        "APS": (mids, edges, y, sigma),
    }

    monkeypatch.setattr(mp, "load_aufi_oneday", lambda *_args, **_kwargs: {k: v.copy() for k, v in frames.items()})
    monkeypatch.setattr(
        mp,
        "read_inlet_flag",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"InletFlag_LARGE": np.zeros(120, dtype=int)},
            index=pd.date_range("2024-05-28 00:00:00", periods=120, freq="s"),
        ),
    )
    monkeypatch.setattr(
        mp,
        "read_microphysical",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"CNgt10nm": np.ones(120)},
            index=pd.date_range("2024-05-28 00:00:00", periods=120, freq="s"),
        ),
    )
    monkeypatch.setattr(mp, "plot_period_totals", lambda *_args, **_kwargs: plt.subplots())
    monkeypatch.setattr(mp, "chunk_is_incloud", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        mp,
        "make_filtered_specs",
        lambda *_args, **_kwargs: (
            dict(base_specs),
            {},
            {},
            {name: np.array([10, 10]) for name in base_specs},
        ),
    )
    monkeypatch.setattr(mp, "run_joint_optimization", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")))

    try:
        mp.run_arcsix_merge_for_periods(
            [("2024-05-28 00:00:00", "2024-05-28 00:00:59")],
            tmp_path / "data",
            tmp_path / "out",
            fims_lag=0,
            min_samples_per_inst=1,
        )
    except RuntimeError as exc:
        assert "1 ARCSIX merge chunk" in str(exc)
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected failed chunk to raise after logging")


def test_fims_aps_only_writes_selected_fims_spectrum(monkeypatch, tmp_path):
    sys.path.insert(0, str(SRC))
    import matplotlib.pyplot as plt
    import pandas as pd
    from sizedistmerge import arcsix_merge_production as mp

    idx = pd.date_range("2024-05-28 00:00:00.500", periods=2, freq="30s")
    frames = {
        name: pd.DataFrame({"placeholder": np.ones(len(idx))}, index=idx)
        for name in ("APS", "FIMS")
    }
    fims_edges = np.array([10.0, 20.0, 40.0, 80.0])
    fims_mids = np.sqrt(fims_edges[:-1] * fims_edges[1:])
    fims_y = np.array([1.0, 2.0, 99.0])
    fims_sigma = np.array([0.1, 0.2, 9.9])
    aps_edges = np.array([40.0, 80.0, 160.0])
    aps_mids = np.sqrt(aps_edges[:-1] * aps_edges[1:])
    aps_y = np.array([4.0, 5.0])
    aps_sigma = np.array([0.4, 0.5])
    captured = {}
    written = {}

    monkeypatch.setattr(mp, "load_af_oneday", lambda *_args, **_kwargs: {k: v.copy() for k, v in frames.items()})
    monkeypatch.setattr(
        mp,
        "read_inlet_flag",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"InletFlag_LARGE": np.zeros(120, dtype=int)},
            index=pd.date_range("2024-05-28 00:00:00", periods=120, freq="s"),
        ),
    )
    monkeypatch.setattr(
        mp,
        "read_microphysical",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"CNgt10nm": np.ones(120)},
            index=pd.date_range("2024-05-28 00:00:00", periods=120, freq="s"),
        ),
    )
    monkeypatch.setattr(mp, "plot_period_totals", lambda *_args, **_kwargs: plt.subplots())
    monkeypatch.setattr(mp, "chunk_is_incloud", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        mp,
        "make_filtered_specs",
        lambda *_args, **_kwargs: (
            {
                "FIMS": (fims_mids, fims_edges, fims_y, fims_sigma),
                "APS": (aps_mids, aps_edges, aps_y, aps_sigma),
            },
            {},
            {},
            {"FIMS": np.array([10, 10, 10]), "APS": np.array([10, 10])},
        ),
    )

    def fake_make_tikhonov_merged_spec(**kwargs):
        captured["e_fims_sel"] = kwargs["e_fims_sel"]
        captured["y_fims_sel"] = kwargs["y_fims_sel"]
        return (
            {"merged": (fims_mids[:2], fims_edges[:3], np.array([7.0, 8.0]), np.array([0.7, 0.8]))},
            {},
            {},
            {},
        )

    monkeypatch.setattr(mp, "make_tikhonov_merged_spec", fake_make_tikhonov_merged_spec)
    monkeypatch.setattr(mp, "plot_sizedist_all", lambda **_kwargs: (plt.subplots(), plt.subplots(), ({}, {})))
    monkeypatch.setattr(mp, "remap_dndlog_by_edges_any", lambda _old, _new, vals: np.asarray(vals, float))

    def fake_write_day_netcdf(*_args, **kwargs):
        written.update(kwargs)
        return tmp_path / "out.nc"

    monkeypatch.setattr(mp, "write_day_netcdf", fake_write_day_netcdf)

    mp.run_arcsix_merge_for_periods(
        [("2024-05-28 00:00:00", "2024-05-28 00:00:59")],
        tmp_path / "data",
        tmp_path / "out",
        instruments=("FIMS", "APS"),
        apply_alignment=False,
        fims_lag=0,
        min_samples_per_inst=1,
        fims_xmax=40.0,
    )

    assert captured["e_fims_sel"].tolist() == [10.0, 20.0, 40.0]
    assert captured["y_fims_sel"].tolist() == [1.0, 2.0]
    assert written["day_fims_algn"].tolist() == [[1.0, 2.0]]


def test_write_day_netcdf_uses_canonical_name_and_pops_variables(tmp_path):
    sys.path.insert(0, str(SRC))
    from netCDF4 import Dataset
    from sizedistmerge import arcsix_merge_production as mp

    edges = np.array([10.0, 20.0, 40.0])
    vals = np.array([[1.0, 2.0]])

    nc_path = mp.write_day_netcdf(
        tmp_path,
        "2024-05-28",
        day_fine_edges=edges,
        day_fims_algn=vals,
        day_uhsas_algn=vals + 1.0,
        day_pops_algn=vals + 2.0,
        day_aps_algn=vals + 3.0,
        day_fine_vals=vals + 4.0,
        day_times_start=[np.datetime64("2024-05-28T00:00:00")],
        day_times_end=[np.datetime64("2024-05-28T00:01:00")],
        day_incloud_flag=[0],
        day_n_fit=[1.45],
        day_n_pops_fit=[1.55],
        day_rho_fit=[1200.0],
        day_best_cost=[0.25],
        orig_APS_edges=edges,
        orig_UHSAS_edges=edges,
        orig_POPS_edges=edges,
        orig_FIMS_edges=edges,
    )

    assert nc_path.name == "2024-05-28_sizedist_merged.nc"
    assert "_v2" not in nc_path.name

    with Dataset(nc_path) as nc:
        assert "pops_aligned_dNdlogDp" in nc.variables
        assert "retrieved_pops_n_fit" in nc.variables
        assert "pops_edges_nm" in nc.variables
        assert nc.variables["retrieved_pops_n_fit"][:].tolist() == [1.55]
        assert "FIMS-UHSAS-POPS-APS" in nc.description


def test_write_day_netcdf_omits_pops_variables_when_pops_is_absent(tmp_path):
    sys.path.insert(0, str(SRC))
    from netCDF4 import Dataset
    from sizedistmerge import arcsix_merge_production as mp

    edges = np.array([10.0, 20.0, 40.0])
    vals = np.array([[1.0, 2.0]])

    nc_path = mp.write_day_netcdf(
        tmp_path,
        "2024-05-28",
        day_fine_edges=edges,
        day_fims_algn=vals,
        day_uhsas_algn=vals + 1.0,
        day_aps_algn=vals + 3.0,
        day_fine_vals=vals + 4.0,
        day_times_start=[np.datetime64("2024-05-28T00:00:00")],
        day_times_end=[np.datetime64("2024-05-28T00:01:00")],
        day_incloud_flag=[0],
        day_n_fit=[1.45],
        day_rho_fit=[1200.0],
        day_best_cost=[0.25],
        orig_APS_edges=edges,
        orig_UHSAS_edges=edges,
        orig_FIMS_edges=edges,
    )

    with Dataset(nc_path) as nc:
        assert "pops_aligned_dNdlogDp" not in nc.variables
        assert "retrieved_pops_n_fit" not in nc.variables
        assert "pops_edges_nm" not in nc.variables
        assert "FIMS-UHSAS-APS" in nc.description


def test_post_merge_qc_and_icartt_conversion_use_packaged_workflow(monkeypatch, tmp_path):
    sys.path.insert(0, str(SRC))
    import pandas as pd
    from netCDF4 import Dataset
    from sizedistmerge import arcsix_merge_production as mp

    day_dir = tmp_path / "merge" / "2024-05-28"
    day_dir.mkdir(parents=True)
    edges = np.array([10.0, 100.0])
    raw_vals = np.array([[100.0], [110.0], [600.0]])
    starts = pd.to_datetime(
        [
            "2024-05-28 00:00:00",
            "2024-05-28 00:01:00",
            "2024-05-28 00:02:00",
        ]
    )
    ends = starts + pd.Timedelta(seconds=59)

    mp.write_day_netcdf(
        day_dir,
        "2024-05-28",
        day_fine_edges=edges,
        day_fims_algn=raw_vals,
        day_uhsas_algn=raw_vals,
        day_pops_algn=raw_vals,
        day_aps_algn=raw_vals,
        day_fine_vals=raw_vals,
        day_times_start=starts,
        day_times_end=ends,
        day_incloud_flag=[0, 0, 0],
        day_n_fit=[1.45, 1.46, 1.47],
        day_n_pops_fit=[1.55, 1.56, 1.57],
        day_rho_fit=[1000.0, 1010.0, 1020.0],
        day_best_cost=[0.1, 0.3, 0.4],
        orig_APS_edges=edges,
        orig_UHSAS_edges=edges,
        orig_POPS_edges=edges,
        orig_FIMS_edges=edges,
    )

    cpc_index = pd.date_range("2024-05-28 00:00:00", periods=240, freq="s")
    cpc = pd.DataFrame({"CNgt10nm": np.full(cpc_index.size, 100.0)}, index=cpc_index)
    monkeypatch.setattr(mp, "read_microphysical", lambda *_args, **_kwargs: cpc)

    qc = mp.run_post_merge_product_qc(
        tmp_path / "merge",
        tmp_path / "data",
        min_points_for_robust=3,
        k_sigma_warn=2.0,
        k_sigma_drop=10.0,
        plot_hi=700.0,
    )

    assert qc.total_chunks == 3
    assert qc.kept_chunks == 2
    assert qc.dropped_extreme_chunks == 1
    assert qc.qc_table_path.exists()
    assert (qc.qc_plot_dir / "hist_optimization_best_cost_with_thresh.png").exists()
    assert len(qc.qc_netcdf_paths) == 1

    table = pd.read_csv(qc.qc_table_path)
    assert table["warning_high_cost"].tolist() == [0, 1, 1]
    assert table["warning_merged_gt10_diff_from_cpc"].tolist() == [0, 0, 1]

    with Dataset(qc.qc_netcdf_paths[0]) as nc:
        assert len(nc.dimensions["chunk"]) == 2
        assert "warning_high_cost" in nc.variables
        assert "warning_merged_gt10_diff_from_cpc" in nc.variables
        assert nc.variables["warning_high_cost"][:].tolist() == [0, 1]

    ict_paths = mp.convert_qc_netcdf_to_icartt(
        qc.qc_netcdf_dir,
        tmp_path / "merge" / "icartt_from_qc_flagged_nc",
        product_name="ARCSIX-MERGED-SIZEDIST",
        revision="R1",
    )
    assert len(ict_paths) == 1
    assert ict_paths[0].name == "ARCSIX-MERGED-SIZEDIST_P3B_20240528_R1.ict"

    lines = ict_paths[0].read_text().splitlines()
    nheader = int(lines[0].split(",", 1)[0])
    assert nheader < len(lines)
    assert "1001" in lines[0]
    assert any("Time_Start" in line for line in lines[:nheader])
    assert any("warning_merged_gt10_diff_from_cpc" in line for line in lines[:nheader])
    assert not any("quadlog" in line.lower() for line in lines[:nheader])
    assert any("linear residual outlier" in line for line in lines[:nheader])
    assert len(lines[nheader:]) == 2
