"""Resource path helpers for packaged SizeDistMerge data."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

_LUT_NAMES = {
    "uhsas": "uhsas_sigma_col_1054nm.zarr",
    "pops": "pops_sigma_col_405nm.zarr",
    "pcasp": "pcasp_sigma_col_632p8nm.zarr",
}


def _source_tree_lut_root() -> Path:
    return Path(__file__).resolve().parents[2] / "lut"


def _normalize_lut_kind(kind: str) -> str:
    if not isinstance(kind, str):
        raise TypeError("kind must be a string")
    key = kind.strip().lower()
    if not key:
        raise ValueError("kind must not be empty")
    return key


def lut_path(kind: str) -> Path:
    """Return the POPS, UHSAS, or PCASP LUT directory path.

    Parameters
    ----------
    kind
        One of ``"uhsas"``, ``"pops"``, or ``"pcasp"``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown.
    FileNotFoundError
        If the requested LUT is not available as package data or in the
        source-tree ``lut/`` directory.
    """
    key = _normalize_lut_kind(kind)
    try:
        name = _LUT_NAMES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_LUT_NAMES))
        raise ValueError(f"unknown LUT kind {kind!r}; expected one of: {allowed}") from exc

    source_tree = _source_tree_lut_root() / name
    # In a checkout, use the single authoritative top-level LUT directory.
    if (Path(__file__).resolve().parents[2] / "pyproject.toml").is_file() and source_tree.is_dir():
        return source_tree

    # setuptools packages that same directory as sizedistmerge.lut at install time.
    try:
        packaged = resources.files("sizedistmerge.lut").joinpath(name)
    except ModuleNotFoundError:
        packaged = None
    if packaged is not None and packaged.is_dir():
        return Path(packaged)

    raise FileNotFoundError(
        f"{name} was not found in package data or at {source_tree}. "
        "Install package data or pass an explicit lut_dir to the merge pipeline."
    )


__all__ = ["lut_path"]
