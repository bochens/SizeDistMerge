# src/ict_utility.py
# Generic ICARTT/ICT utilities — identical functionality to your previous module,
# but with "arcsix" removed from names and text.
#
# Highlights:
# - read_ict_file / read_ict / read_ict_dir
# - pick_ict_files (optional "prefix" lets you match e.g. "ARCSIX-")
# - Instrument readers: read_aps / read_pops / read_uhsas / read_fims / read_nmass / read_inlet_flag
# - Bin metadata extraction, relabeling, spectra helpers, grid checks, alignment, filtering,
#   flag analysis — all preserved.

import re
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from io import StringIO

__all__ = [
    # IO
    "read_ict",
    "read_ict_file",
    "read_ict_dir",
    "pick_ict_files",
    "parse_bound",
    # Instrument wrappers
    "read_aps",
    "read_pops",
    "read_uhsas",
    "read_fims",
    "read_nmass",
    "read_inlet_flag",
    "read_microphysical",
    # Bins / spectra
    "label_size_bins",
    "get_spectra",
    "mean_spectrum",
    # Time / grids
    "check_common_grid",
    "align_to_common_grid",
    "filter_by_spectra_presence",
    # Flags
    "find_flag_column",
    "flag_segments",
    "flag_fractions",
    # Debug helper
    "check_meta",
]

PathLike = Union[str, Path]

# ─────────────────────────────── Small helpers ───────────────────────────────

def _instrument_from_filename(name: str) -> Optional[str]:
    """
    Generic instrument extraction:
    - Takes the token before the first '_' (e.g. 'ARCSIX-LARGE-APS' or 'PUTLS-UHSAS')
    - Returns the part after the first '-' if present, else the token itself
      -> 'ARCSIX-LARGE-APS' -> 'LARGE-APS'
      -> 'PUTLS-UHSAS'      -> 'PUTLS-UHSAS'
      -> 'FIMS'             -> 'FIMS'
    """
    head = name.split("_", 1)[0]
    return head.split("-", 1)[1] if "-" in head else head

def parse_bound(x: Optional[Union[str, pd.Timestamp]], tz: str = "UTC") -> Optional[pd.Timestamp]:
    """Parse a start/end bound into UTC."""
    if x is None:
        return None
    ts = pd.to_datetime(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts.tz_convert("UTC")

def _has_time_component(x: Optional[Union[str, pd.Timestamp]]) -> bool:
    """True if input includes a time-of-day (not just a date)."""
    if x is None:
        return False
    if isinstance(x, str):
        return bool(re.search(r"\d{1,2}:\d{2}", x))
    ts = pd.to_datetime(x)
    return any([ts.hour, ts.minute, ts.second, ts.microsecond])

def _local_day_bounds(x: Union[str, pd.Timestamp], tz: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return [day_start_local, day_end_local) converted to UTC."""
    ts = pd.to_datetime(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    day0_local = ts.normalize()
    day1_local = day0_local + pd.Timedelta(days=1)
    return day0_local.tz_convert("UTC"), day1_local.tz_convert("UTC")

def _ensure_time_column(df: pd.DataFrame, *, want_column: bool) -> pd.DataFrame:
    """Ensure time is present as a column (want_column=True) or index (want_column=False)."""
    if want_column:
        if "time" in df.columns:
            return df
        if isinstance(df.index, pd.DatetimeIndex):
            out = df.reset_index()
            if out.columns[0] != "time":
                out = out.rename(columns={out.columns[0]: "time"})
            return out
        return df
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        if "time" in df.columns:
            return df.set_index("time")
        return df

# ─────────────────────────────── Table parsing ───────────────────────────────

def _infer_meas_date(lines: List[str], path: Path) -> Optional[datetime]:
    """From 6-int header line or any _YYYYMMDD_ in filename/header."""
    for s in lines[:300]:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 6 and all(p.isdigit() for p in parts):
            y1, m1, d1, *_ = map(int, parts)
            try:
                return datetime(y1, m1, d1, tzinfo=timezone.utc)
            except Exception:
                pass
    m = re.search(r"(\d{8})", path.name)
    if m:
        ymd = m.group(1)
        y, mo, d = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc)
        except Exception:
            pass
    for s in lines[:300]:
        m = re.search(r"(\d{8})", s)
        if m:
            ymd = m.group(1)
            y, mo, d = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])
            try:
                return datetime(y, mo, d, tzinfo=timezone.utc)
            except Exception:
                pass
    return None

def _read_table(lines: List[str]) -> pd.DataFrame:
    """
    Find the last 'Time_*,...' header (handles Time_Mid, Time_Start) and read the CSV below it.
    Works for files that repeat 'Time_*' earlier in the ICARTT header.
    """
    header_idx = None
    for i, s in enumerate(lines):
        if s.startswith("Time_") or s.startswith("Time,") or s.startswith("UTC,"):
            header_idx = i  # keep last
    if header_idx is None:
        # fallback: last comma-delimited line
        for i, s in enumerate(lines):
            if "," in s:
                header_idx = i
        if header_idx is None:
            raise ValueError("Could not find a comma-delimited header line.")
    cols = [c.strip() for c in lines[header_idx].split(",")]
    body = "\n".join(lines[header_idx + 1:]).strip()
    if not body:
        raise ValueError("No data rows found after the header line.")
    df = pd.read_csv(StringIO(body), names=cols, engine="python", on_bad_lines="skip")
    return df

def _attach_time(df: pd.DataFrame, meas_date: Optional[datetime]) -> pd.DataFrame:
    """
    Build UTC timestamps from Time_Mid or Time_Start with rollover handling.
    Replace ICARTT sentinels with NaN (clean time column first).
    """
    # Choose time column
    if "Time_Mid" in df.columns:
        tcol = "Time_Mid"
    elif "Time_Start" in df.columns:
        tcol = "Time_Start"
    elif "Time" in df.columns:
        tcol = "Time"
    elif "UTC" in df.columns:
        tcol = "UTC"
    else:
        return df

    # Clean time column first
    tvals = pd.to_numeric(df[tcol], errors="coerce")
    tvals = tvals.mask(tvals.isin([-9999.0, -8888.0, -7777.0]))
    df[tcol] = tvals
    if tvals.notna().sum() == 0:
        return df

    # Replace sentinels elsewhere
    df = df.replace((-9999, -7777, -8888), np.nan)

    secs = df[tcol].astype(float).to_numpy()

    # Epoch seconds?
    if np.nanmax(secs) >= 1e9:
        times = pd.to_datetime(secs, unit="s", utc=True)
        df.insert(0, "time", times)
        return df

    # Need a date for seconds-of-day
    if meas_date is None:
        warnings.warn("No measurement date found; keeping numeric time column.", RuntimeWarning)
        return df

    rolls = np.r_[0, (np.diff(secs) < 0).astype(int)].cumsum()
    times = [meas_date + timedelta(days=int(rolls[i]), seconds=float(secs[i])) for i in range(secs.size)]
    df.insert(0, "time", pd.to_datetime(times, utc=True))
    return df

# ────────────────────── Bin metadata parsing & renaming ──────────────────────

_NUM_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def _parse_num_list(s: str) -> List[float]:
    """Extract floats from a string, ignoring obviously bogus huge numbers."""
    nums = []
    for token in _NUM_RE.findall(s):
        try:
            val = float(token)
        except Exception:
            continue
        if val > 1e5:
            continue
        nums.append(val)
    return nums

def _parse_bracketed_num_list(s: str) -> Optional[List[float]]:
    """Prefer list inside the first [...] if present."""
    bracket_match = re.search(r'\[([^\]]+)\]', s)
    if not bracket_match:
        return None
    return _parse_num_list(bracket_match.group(1))

def _read_list_after_marker(lines: List[str], i: int) -> Optional[List[float]]:
    """
    Read a numeric list that may start on the same line as the marker
    or wrap onto the next few lines.
    """
    buf = ""
    for j in range(i, min(i + 10, len(lines))):
        s = lines[j].strip()
        buf += " " + s
        if ']' in s:
            bracket_match = re.search(r'\[([^\]]+)\]', buf)
            if bracket_match:
                return _parse_num_list(bracket_match.group(1))
            break
    return _parse_num_list(lines[i])

def _parse_geometric_mean_line(s: str) -> List[float]:
    """
    For FIMS-style: extract mids from a sentence like
      'Geometric mean particle diameters ... size bins 1-30 are 10.0, ..., 600.0. Size distribution dlogD = 0.061'
    We take numbers AFTER ' are ' and BEFORE any trailing 'size distribution' / 'dlog' text.
    Also drop tiny stray ints (e.g., '1', '30') from '1-30'.
    """
    sl = s.lower()

    # end before trailing info
    end = len(s)
    for marker in ("size distribution", "dlogd", "dlogdp"):
        k = sl.find(marker)
        if k != -1:
            end = min(end, k)

    # start after ' are ' if present; else from beginning
    start = sl.find(" are ")
    segment = s[start + 5:end] if start != -1 else s[:end]

    nums = _parse_num_list(segment)
    # keep mids >= ~5 nm
    return [x for x in nums if x >= 5]

def _extract_bin_meta_from_header(lines: List[str], instrument_hint: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Extract bin metadata for APS/POPS/UHSAS/FIMS and NMASS cut sizes from header text.
    Returns keys among: lower_nm, mid_nm, upper_nm, dlogD, nmass_cut_nm
    """
    meta: Dict[str, List[float]] = {}
    n = len(lines)
    low = lambda s: s.lower()

    # -------- FIMS --------
    if instrument_hint in (None, "FIMS") or not meta:
        for i, s in enumerate(lines):
            sl = low(s)
            if ("geometric" in sl and "mean" in sl and "diameter" in sl and "nm" in sl) and ("mid_nm" not in meta):
                mids = _parse_geometric_mean_line(s)
                if mids:
                    meta["mid_nm"] = mids
            if ("dlogdp" in sl or "dlogd" in sl) and ("dlogD" not in meta):
                nums = _parse_num_list(s)
                dlog = [v for v in nums if 0 < v < 1]
                if dlog:
                    meta["dlogD"] = [dlog[0]]
            if ("mobility" in sl or "fims" in sl) and ("lower" in sl and "nm" in sl) and ("lower_nm" not in meta):
                nums = _read_list_after_marker(lines, i)
                if nums: meta["lower_nm"] = nums
            if ("mobility" in sl or "fims" in sl) and ("upper" in sl and "nm" in sl) and ("upper_nm" not in meta):
                nums = _read_list_after_marker(lines, i)
                if nums: meta["upper_nm"] = nums

    # -------- POPS --------
    if instrument_hint == "PUTLS-POPS" or any("pops" in low(s) for s in lines[:80]):
        for i, s in enumerate(lines[:600]):
            sl = low(s)
            if "dlogd" in sl:
                nums = _read_list_after_marker(lines, i)
                if nums: meta["dlogD"] = [v for v in nums if 0 < v < 1]
            elif "bin lower-bound diameters" in sl and "nm" in sl:
                nums = _read_list_after_marker(lines, i)
                if nums: meta["lower_nm"] = nums
            elif "bin upper-bound diameters" in sl and "nm" in sl:
                nums = _read_list_after_marker(lines, i)
                if nums: meta["upper_nm"] = nums
            elif (("bin mid-point diameters" in sl and "nm" in sl) or ("mid-point" in sl and "nm" in sl)):
                nums = _read_list_after_marker(lines, i)
                if nums: meta["mid_nm"] = nums

    # -------- APS --------
    if instrument_hint in (None, "LARGE-APS") or not meta:
        for i, s in enumerate(lines):
            if "bin parameters (in nm) are" in low(s):
                for j in range(i + 1, min(i + 7, n)):
                    t = lines[j]; tl = low(t)
                    nums = _parse_bracketed_num_list(t) or _parse_num_list(t)
                    if "lower" in tl and "bound" in tl and nums:
                        meta.setdefault("lower_nm", nums)
                    elif ("mid points" in tl or ("mid" in tl and "point" in tl)) and nums:
                        meta.setdefault("mid_nm", nums)
                    elif "upper" in tl and "bound" in tl and nums:
                        meta.setdefault("upper_nm", nums)

    # -------- UHSAS --------
    if instrument_hint in (None, "PUTLS-UHSAS") or not meta:
        for i, s in enumerate(lines):
            sl = low(s)
            if "bin lower-bound diameters" in sl and "nm" in sl and "lower_nm" not in meta:
                nums = _read_list_after_marker(lines, i)
                if nums: meta["lower_nm"] = nums
            elif "bin upper-bound diameters" in sl and "nm" in sl and "upper_nm" not in meta:
                nums = _read_list_after_marker(lines, i)
                if nums: meta["upper_nm"] = nums
            elif (("bin mid-point diameters" in sl and "nm" in sl) or ("mid-point" in sl and "nm" in sl)) and "mid_nm" not in meta:
                nums = _read_list_after_marker(lines, i)
                if nums: meta["mid_nm"] = nums
            elif "dlogd" in sl and "dlogD" not in meta:
                nums = _read_list_after_marker(lines, i)
                if nums: meta["dlogD"] = [v for v in nums if 0 < v < 1]

    # -------- NMASS cuts --------
    for s in lines:
        sl = low(s)
        if "nmass" in sl and "cut" in sl:
            nums = _parse_bracketed_num_list(s)
            if nums:
                meta["nmass_cut_nm"] = [float(x) for x in nums]
                break

    # derive mids if missing
    if "mid_nm" not in meta and "lower_nm" in meta and "upper_nm" in meta:
        lo, up = meta["lower_nm"], meta["upper_nm"]
        if len(lo) == len(up):
            meta["mid_nm"] = [float(np.sqrt(l * u)) for l, u in zip(lo, up)]

    return meta

# ─────────────────────────────── File picking ───────────────────────────────

def _dates_from_bounds(start, end, tz="UTC") -> List[str]:
    """Inclusive list of YYYYMMDD (UTC) covered by start..end."""
    t0, t1 = parse_bound(start, tz), parse_bound(end, tz)
    if t0 is None and t1 is None:
        raise ValueError("Provide start or end to infer date(s) when root is a directory.")
    if t0 is None:
        t0 = t1
    if t1 is None:
        t1 = t0
    if t1 < t0:
        t0, t1 = t1, t0
    dates = pd.date_range(t0.normalize(), t1.normalize(), freq="D", tz="UTC")
    return [d.strftime("%Y%m%d") for d in dates]

def _pattern(instrument: Optional[str], *, platform: Optional[str] = "P3B",
             exts: Sequence[str] = ("ict","txt"), prefix: Optional[str] = None) -> re.Pattern:
    """
    Build filename regex. Requires `instrument` when scanning a directory.
    Examples (with prefix="ARCSIX"):
      ARCSIX-LARGE-APS_P3B_YYYYMMDD_R*.ict
      ARCSIX-PUTLS-NMASS_P3B_YYYYMMDD_R*.txt
    If prefix=None, matches like:
      LARGE-APS_P3B_YYYYMMDD_R*.ict
    """
    if not instrument:
        raise ValueError("instrument must be provided when reading from a directory.")
    ext_pat = "|".join(re.escape(e) for e in exts)
    pre = (re.escape(prefix) + r"-") if prefix else ""
    if platform:
        pat = rf'^{pre}{re.escape(instrument)}_{re.escape(platform)}_([0-9]{{8}})_R(\d+)(?:_L(\d+))?\.(?:{ext_pat})$'
    else:
        pat = rf'^{pre}{re.escape(instrument)}_([0-9]{{8}})_R(\d+)(?:_L(\d+))?\.(?:{ext_pat})$'
    return re.compile(pat, re.IGNORECASE)

def pick_ict_files(
    root: PathLike,
    *,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    tz: str = "UTC",
    instrument: Optional[str] = None,   # REQUIRED when root is a directory
    platform: Optional[str] = "P3B",
    exts: Sequence[str] = ("ict", "txt"),
    choose: str = "latest",             # "latest" per day or "all"
    prefix: Optional[str] = None,       # e.g., "ARCSIX"
) -> List[Path]:
    """Return instrument files for day(s) covered by start/end, or accept a full path."""
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"No such file or directory: {root}")
    rx = _pattern(instrument, platform=platform, exts=exts, prefix=prefix)

    days = set(_dates_from_bounds(start, end, tz))
    by_date: Dict[str, List[Tuple[Path, int, int]]] = {d: [] for d in days}

    for p in root.iterdir():
        if not p.is_file():
            continue
        file_match = rx.match(p.name)
        if not file_match:
            continue
        ymd, r, l = file_match.group(1), int(file_match.group(2) or 0), int((file_match.group(3) or 0))
        if ymd in by_date:
            by_date[ymd].append((p, r, l))

    out: List[Path] = []
    for ymd in sorted(days):
        matches = by_date.get(ymd, [])
        if not matches:
            continue
        if choose == "all":
            out.extend(sorted((p for p, _, _ in matches), key=lambda pp: pp.name))
        else:
            out.append(max(matches, key=lambda t: (t[1], t[2], t[0].name))[0])  # highest R, then L

    return sorted(out, key=lambda p: p.name)

# ─────────────────────────────── Readers (generic) ───────────────────────────

def _read_table_with_header(lines: List[str]) -> pd.DataFrame:
    """Internal: identical to _read_table but kept separate for clarity."""
    return _read_table(lines)

def read_ict_file(
    path: PathLike,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    *,
    tz: str = "UTC",
    make_index: bool = True,
    keep_time_col: bool = False,
) -> pd.DataFrame:
    """
    Read one ICT/ICARTT file and filter by start/end.
    Attaches df.attrs['bin_meta'] (from header only).
    """
    path = Path(path)
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()
    if not lines:
        raise ValueError(f"File is empty or unreadable: {path}")

    instr_key = _instrument_from_filename(path.name)
    bin_meta = _extract_bin_meta_from_header(lines, instrument_hint=instr_key)

    # Friendly warnings if expected metadata is missing
    if instr_key in ("LARGE-APS", "PUTLS-POPS", "PUTLS-UHSAS"):
        if not any(k in bin_meta for k in ("mid_nm", "lower_nm", "upper_nm")):
            warnings.warn(f"{instr_key}: No bin metadata found in header; no relabeling will be applied.")
    if instr_key == "PUTLS-NMASS" and "nmass_cut_nm" not in bin_meta:
        warnings.warn("NMASS cut sizes not found in header; CNch* columns will not be renamed.")

    meas_date = _infer_meas_date(lines, path)
    df = _attach_time(_read_table_with_header(lines), meas_date)
    df.attrs["instrument"] = instr_key

    # Time subsetting
    t0_utc = parse_bound(start, tz); t1_utc = parse_bound(end, tz)
    single_day = ((start is None) != (end is None)) or (t0_utc is not None and t1_utc is not None and t0_utc == t1_utc)
    if "time" in df.columns:
        if single_day:
            base = start if start is not None else end
            day0, day1 = _local_day_bounds(base, tz)
            df = df[(df["time"] >= day0) & (df["time"] < day1)]
            if _has_time_component(base) and len(df):
                target_utc = parse_bound(base, tz)
                i = (df["time"] - target_utc).abs().values.argmin()
                df = df.iloc[[i]].copy()
        else:
            if t0_utc is not None: df = df[df["time"] >= t0_utc]
            if t1_utc is not None: df = df[df["time"] < t1_utc]

    if make_index and "time" in df.columns:
        df = df.set_index("time")

    if not keep_time_col:
        for c in ("Time_Mid", "Time_Start"):
            if c in df.columns:
                df = df.drop(columns=[c])
                break

    # Attach parsed metadata; no bin renaming here (standardization happens in read_ict/_read_multi if requested)
    if bin_meta:
        df.attrs["bin_meta"] = bin_meta

        # NMASS channel rename (no rounding; keep decimals)
        if instr_key == "PUTLS-NMASS" and "nmass_cut_nm" in bin_meta:
            df = _rename_nmass_channels(df, bin_meta["nmass_cut_nm"])

    return df

def _read_multi(
    files: List[Path],
    *,
    start: Optional[Union[str, pd.Timestamp]],
    end: Optional[Union[str, pd.Timestamp]],
    tz: str,
    keep_time_col: bool,
    make_index: bool = True,
    standardize_bins: bool = False,
    col_prefix: str = "dNdlogDp",
) -> pd.DataFrame:
    """Read multiple ICT files, concat and sort by time. Can standardize bin columns."""
    parts: List[pd.DataFrame] = []
    for fp in files:
        p = read_ict_file(fp, start=start, end=end, tz=tz,
                          make_index=False, keep_time_col=True)
        if not len(p):
            continue

        instr = (p.attrs.get("instrument") or "").upper()
        if standardize_bins and instr not in ("PUTLS-NMASS", "LARGE-INLETFLAG"):
            p = label_size_bins(p, col_prefix=col_prefix)

        p = _ensure_time_column(p, want_column=True)
        parts.append(p)

    if not parts:
        empty = pd.DataFrame(columns=["time"])
        return empty.set_index("time") if make_index else empty

    out = pd.concat(parts, ignore_index=True)

    if "time" in out.columns:
        out = out.sort_values("time")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()

    if make_index:
        out = _ensure_time_column(out, want_column=False)
    else:
        out = _ensure_time_column(out, want_column=True)

    if getattr(parts[0], "attrs", None):
        out.attrs.update(parts[0].attrs)

    return out

def read_ict(
    path_or_dir: PathLike,
    *,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    tz: str = "UTC",
    instrument: Optional[str] = None,
    platform: Optional[str] = "P3B",
    exts: Sequence[str] = ("ict", "txt"),
    choose: str = "latest",
    make_index: bool = True,
    keep_time_col: bool = False,
    standardize_bins: bool = False,
    col_prefix: str = "dNdlogDp",
    prefix: Optional[str] = None,  # e.g., "ARCSIX"
) -> pd.DataFrame:
    """
    Generic ICT reader (file or dir) using ONLY header metadata.
    If standardize_bins=True and instrument is APS/POPS/UHSAS/FIMS,
    size-bin columns are relabeled to <col_prefix>_d<mid>nm (no rounding).
    """
    root = Path(path_or_dir)
    if root.is_file():
        df = read_ict_file(root, start=start, end=end, tz=tz,
                           make_index=False, keep_time_col=True)
        if len(df):
            instr = (df.attrs.get("instrument") or "").upper()
            if standardize_bins and instr not in ("PUTLS-NMASS", "LARGE-INLETFLAG"):
                df = label_size_bins(df, col_prefix=col_prefix)
        df = _ensure_time_column(df, want_column=not make_index)
        if "time" in df.columns:
            df = df.sort_values("time")
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        if make_index:
            df = _ensure_time_column(df, want_column=False)
        if not keep_time_col:
            for c in ("Time_Mid", "Time_Start"):
                if c in df.columns:
                    df = df.drop(columns=[c]); break
        return df

    files = pick_ict_files(root, start=start, end=end, tz=tz,
                           instrument=instrument, platform=platform,
                           exts=exts, choose=choose, prefix=prefix)
    if not files:
        raise FileNotFoundError(f"No {instrument or '<instrument>'} files found in {root} for the requested time range.")
    df = _read_multi(files, start=start, end=end, tz=tz,
                     keep_time_col=True, make_index=False,
                     standardize_bins=standardize_bins,
                     col_prefix=col_prefix)
    df = _ensure_time_column(df, want_column=not make_index)
    if "time" in df.columns:
        df = df.sort_values("time")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    if make_index:
        df = _ensure_time_column(df, want_column=False)
    if not keep_time_col:
        if "Time_Mid" in df.columns:   df = df.drop(columns=["Time_Mid"])
        if "Time_Start" in df.columns: df = df.drop(columns=["Time_Start"])
    return df

def read_ict_dir(
    root: PathLike,
    pattern: str = "*.ict",
    **kwargs,
) -> pd.DataFrame:
    """Read all files matching `pattern` in a directory and concatenate."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No such directory: {root}")
    files = sorted(root.glob(pattern))
    if not files:
        return pd.DataFrame()

    # Defaults consistent with read_ict
    start = kwargs.pop("start", None)
    end = kwargs.pop("end", None)
    tz = kwargs.pop("tz", "UTC")
    keep_time_col = kwargs.pop("keep_time_col", False)
    make_index = kwargs.pop("make_index", True)
    standardize_bins = kwargs.pop("standardize_bins", False)
    col_prefix = kwargs.pop("col_prefix", "dNdlogDp")

    df = _read_multi(
        files,
        start=start,
        end=end,
        tz=tz,
        keep_time_col=True,         # keep for now, drop below if requested
        make_index=False,
        standardize_bins=standardize_bins,
        col_prefix=col_prefix,
    )
    df = _ensure_time_column(df, want_column=not make_index)
    if "time" in df.columns:
        df = df.sort_values("time")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    if make_index:
        df = _ensure_time_column(df, want_column=False)
    if not keep_time_col:
        for c in ("Time_Mid", "Time_Start"):
            if c in df.columns:
                df = df.drop(columns=[c])
                break
    return df

# ─────────────────────────────── Spectra APIs ───────────────────────────────

def get_spectra(
    df: pd.DataFrame,
    *,
    col_prefix: str = "dNdlogDp",
    strict: bool = False,
    require_monotonic: bool = True,
    long: bool = False,
) -> Union[Tuple[np.ndarray, pd.DataFrame], pd.DataFrame]:
    """
    Return the time-resolved spectra without averaging.

    If long=False (default):
        -> (diam_nm, spectra_wide)
           where `diam_nm` is a 1D np.ndarray of mid diameters (nm),
           and `spectra_wide` is a DataFrame indexed by time with columns=diam_nm (floats, nm)

    If long=True:
        -> spectra_long: tidy DataFrame with columns [time, Dp_nm, value] (value = dNdlogDp)
    """
    df_labeled = label_size_bins(
        df,
        col_prefix=col_prefix,
        strict=strict,
        require_monotonic=require_monotonic,
    )

    pattern = re.compile(rf"^{re.escape(col_prefix)}_d(\d+(?:\.\d+)?)nm$", re.IGNORECASE)
    size_cols = [c for c in df_labeled.columns if pattern.match(c)]
    if not size_cols:
        if long:
            return pd.DataFrame(columns=["time", "Dp_nm", "value"])
        return np.array([]), pd.DataFrame(index=df_labeled.index)

    diam_nm = np.array([float(pattern.match(c).group(1)) for c in size_cols])
    order = np.argsort(diam_nm)
    diam_nm = diam_nm[order]
    size_cols = [size_cols[i] for i in order]

    spectra_wide = df_labeled[size_cols].copy()
    spectra_wide.columns = pd.Index(diam_nm, name="Dp_nm")

    if not long:
        return diam_nm, spectra_wide

    spectra_long = (
        spectra_wide
        .stack()
        .to_frame(col_prefix)
        .rename_axis(index=["time", "Dp_nm"])
        .reset_index()
        .rename(columns={col_prefix: "value"})
    )
    return spectra_long

# ──────────────────────── Bin detection and labeling ─────────────────────────

_FIMS_PAT = re.compile(r'^n[_\-]?Dp[_\-]?(\d+)$', re.IGNORECASE)
_RENAMED_PAT = re.compile(r'_d(\d+(?:\.\d+)?)nm$', re.IGNORECASE)

def _detect_bin_columns_any(df: pd.DataFrame):
    cols = list(df.columns)

    # already-renamed: *_d<mid>nm
    renamed: List[Tuple[str, float]] = []
    for col in cols:
        match_obj = _RENAMED_PAT.search(col)
        if match_obj:
            renamed.append((col, float(match_obj.group(1))))
    if renamed:
        renamed.sort(key=lambda t: t[1])
        return "renamed", [c for c, _ in renamed], [mid for _, mid in renamed]

    # APS/POPS/UHSAS style: *_BinNN
    bincols: List[Tuple[int, str]] = []
    pat = re.compile(r'^(?P<prefix>[A-Za-z0-9]+)[_\-]?[Bb]in_?(\d+)$', re.IGNORECASE)
    for col in cols:
        match_obj = pat.match(col)
        if match_obj:
            idx = int(match_obj.group(2))
            bincols.append((idx, col))
    if bincols:
        bincols.sort(key=lambda t: t[0])
        return "bin", [c for _, c in bincols], None

    # FIMS: n_Dp_<n>
    fims: List[Tuple[int, str]] = []
    for col in cols:
        match_obj = _FIMS_PAT.match(col)
        if match_obj:
            idx = int(match_obj.group(1))
            fims.append((idx, col))
    if fims:
        fims.sort(key=lambda t: t[0])
        return "fims", [c for _, c in fims], None

    return "", [], None

def _fmt_num(x: float) -> str:
    """Format numbers for column names without scientific notation and without trailing zeros."""
    s = f"{float(x):.12f}".rstrip("0").rstrip(".")
    return s if s else "0"

def label_size_bins(
    df: pd.DataFrame,
    *,
    col_prefix: str = "dNdlogDp",
    strict: bool = False,
    require_monotonic: bool = True,
) -> pd.DataFrame:
    """
    Relabel bin columns to <col_prefix>_d<middle>nm using the exact mids from header
    (no rounding). Works for APS/POPS/UHSAS/FIMS styles and already-renamed columns.
    """
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    flavor, bin_cols, mids_in_names = _detect_bin_columns_any(df)
    if not bin_cols:
        return df.copy()

    nb = len(bin_cols)
    mids = None

    # 1) mids embedded in names
    if mids_in_names is not None:
        mids = np.asarray(mids_in_names, dtype=float)

    # 2) header meta
    if mids is None:
        meta = getattr(df, "attrs", {}).get("bin_meta", {}) or {}
        lo = np.asarray(meta.get("lower_nm", []), dtype=float) if "lower_nm" in meta else None
        up = np.asarray(meta.get("upper_nm", []), dtype=float) if "upper_nm" in meta else None
        mid_hdr = np.asarray(meta.get("mid_nm", []), dtype=float) if "mid_nm" in meta else None
        dlog_list = meta.get("dlogD", [])
        dlog = float(dlog_list[0]) if isinstance(dlog_list, (list, tuple, np.ndarray)) and len(dlog_list) else None

        if mid_hdr is not None and mid_hdr.size:
            mids = mid_hdr
        if mids is None and lo is not None and up is not None and lo.size and up.size:
            k = min(lo.size, up.size)
            mids = np.sqrt(lo[:k] * up[:k])
        if mids is None and dlog is not None and dlog > 0:
            if lo is not None and lo.size:
                k = min(lo.size, nb)
                mids = lo[:k] * (10.0 ** (dlog / 2.0))
            elif up is not None and up.size:
                k = min(up.size, nb)
                mids = up[:k] / (10.0 ** (dlog / 2.0))

    if mids is None or not np.size(mids):
        return df.copy()

    mids = np.asarray(mids, dtype=float)

    # validate counts & ordering
    if mids.size != nb:
        msg = (f"Size-bin count mismatch: {nb} bin columns vs {mids.size} mids "
               f"(instrument={getattr(df, 'attrs', {}).get('instrument')}).")
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + " Truncating to the shorter length.", RuntimeWarning)

    k = min(nb, mids.size)
    bin_cols = bin_cols[:k]
    mids = mids[:k]

    if require_monotonic and not np.all(np.diff(mids) > 0):
        msg = "Header mids are not strictly increasing; relabeling may misalign with bins."
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)

    out_cols = [f"{col_prefix}_d{_fmt_num(x)}nm" for x in mids]

    non_bin = df.drop(columns=bin_cols, errors="ignore")
    relabeled = pd.DataFrame(df[bin_cols].to_numpy(dtype=float), index=df.index, columns=out_cols)

    out = pd.concat([non_bin, relabeled], axis=1)
    out.attrs.update(getattr(df, "attrs", {}))
    out.attrs["mids_nm"] = mids.tolist()
    return out

def _rename_nmass_channels(df: pd.DataFrame, cuts_nm: List[float]) -> pd.DataFrame:
    """
    Rename CNch1..N -> NMASS_gt<cut>nm using ascending cut sizes (no rounding).
    Only applied if channel count equals len(cuts_nm).
    """
    pat = re.compile(r'^CNch(\d+)$', re.IGNORECASE)
    chans: List[Tuple[str, int]] = []
    for col in df.columns:
        match_obj = pat.match(col)
        if match_obj:
            chans.append((col, int(match_obj.group(1))))
    if not chans:
        return df
    chans_sorted = [c for c, _ in sorted(chans, key=lambda t: t[1])]
    if len(chans_sorted) != len(cuts_nm):
        warnings.warn(f"NMASS rename skipped: {len(chans_sorted)} channels but {len(cuts_nm)} cut sizes.")
        return df
    cuts_sorted = list(sorted(cuts_nm))
    new_names = {c: f"NMASS_gt{_fmt_num(cuts_sorted[i])}nm" for i, c in enumerate(chans_sorted)}
    return df.rename(columns=new_names)

# ───────────────────────────── Convenience wrappers ─────────────────────────

def read_aps(path_or_dir: PathLike, **kwargs) -> pd.DataFrame:
    """APS (aerodynamic). Returns labeled size bins by default."""
    kwargs.setdefault("instrument", "LARGE-APS")
    kwargs.setdefault("standardize_bins", True)
    kwargs.setdefault("col_prefix", "dNdlogDp")
    return read_ict(path_or_dir, **kwargs)

def read_pops(path_or_dir: PathLike, **kwargs) -> pd.DataFrame:
    """POPS (optical PSL). Returns labeled size bins by default."""
    kwargs.setdefault("instrument", "PUTLS-POPS")
    kwargs.setdefault("standardize_bins", True)
    kwargs.setdefault("col_prefix", "dNdlogDp")
    return read_ict(path_or_dir, **kwargs)

def read_uhsas(path_or_dir: PathLike, **kwargs) -> pd.DataFrame:
    """UHSAS ((NH4)2SO4). Returns labeled size bins by default."""
    kwargs.setdefault("instrument", "PUTLS-UHSAS")
    kwargs.setdefault("standardize_bins", True)
    kwargs.setdefault("col_prefix", "dNdlogDp")
    return read_ict(path_or_dir, **kwargs)

def read_fims(path_or_dir: PathLike, **kwargs) -> pd.DataFrame:
    """FIMS (mobility diameters). Returns labeled size bins by default."""
    kwargs.setdefault("instrument", "FIMS")
    kwargs.setdefault("standardize_bins", True)
    kwargs.setdefault("col_prefix", "dNdlogDp")
    return read_ict(path_or_dir, **kwargs)

def read_nmass(path_or_dir: PathLike, **kwargs) -> pd.DataFrame:
    """NMASS (CN channels with 50% cut diameters). No standardization of size bins."""
    kwargs.setdefault("instrument", "PUTLS-NMASS")
    kwargs.setdefault("standardize_bins", False)
    return read_ict(path_or_dir, **kwargs)

def read_inlet_flag(path_or_dir: PathLike, **kwargs) -> pd.DataFrame:
    """InletFlag. No standardization of size bins."""
    kwargs.setdefault("instrument", "LARGE-InletFlag")
    kwargs.setdefault("standardize_bins", False)
    return read_ict(path_or_dir, **kwargs)

def read_microphysical(path_or_dir: PathLike, **kwargs) -> pd.DataFrame:
    """Microphysical (LARGE platform). No bin standardization."""
    kwargs.setdefault("instrument", "LARGE-MICROPHYSICAL")
    kwargs.setdefault("standardize_bins", False)
    return read_ict(path_or_dir, **kwargs)

# ───────────────────────────── Optional utilities ───────────────────────────

def number_to_surface_area_spectrum(mids_nm, dNdlogDp, frac_sigma, unit="um"):
    """
    Convert number spectrum (dN/dlogDp in # cm^-3, D in nm) to surface-area spectrum.
    unit: 'nm' -> nm^2 cm^-3, 'um' -> µm^2 cm^-3, 'm' -> m^2 cm^-3
    Returns: D_out (same unit as chosen for labeling), SA, SA_lo, SA_hi
    """
    D_nm = np.asarray(mids_nm, float)
    y    = np.asarray(dNdlogDp, float)

    if unit == "nm":
        D = D_nm
        area_per_particle = np.pi * (D**2)               # nm^2
        y_unit = r"nm$^2$ cm$^{-3}$"
    elif unit == "um":
        D = D_nm * 1e-3                                  # nm -> µm
        area_per_particle = np.pi * (D**2)               # µm^2
        y_unit = r"$\mu$m$^2$ cm$^{-3}$"
    elif unit == "m":
        D = D_nm * 1e-9                                  # nm -> m
        area_per_particle = np.pi * (D**2)               # m^2
        y_unit = r"m$^2$ cm$^{-3}$"
    else:
        raise ValueError("unit must be 'nm', 'um', or 'm'")

    SA = area_per_particle * y
    lo = SA * (1.0 - frac_sigma)
    hi = SA * (1.0 + frac_sigma)
    return D, SA, lo, hi, y_unit

def mean_spectrum(
    df: pd.DataFrame,
    label: str,
    *,
    col_prefix: str = "dNdlogDp",
    ddof: int = 1,  # sample std by default
):
    """
    Arithmetic time-mean and 1σ (std) for bin-labeled columns like
    "<prefix>_d<mid>nm" (e.g., dNdlogDp_d150nm).

    Returns: (mids_nm, mean_vals, sigma_vals, label) sorted by mids.
    If no matching columns (or df empty), returns None.
    """
    if df is None or df.empty:
        return None

    dnd = label_size_bins(df, col_prefix=col_prefix)

    rx = re.compile(rf"^{re.escape(col_prefix)}_d(\d+(?:\.\d+)?)nm$", re.IGNORECASE)
    cols = [c for c in dnd.columns if rx.match(c)]
    if not cols:
        return None

    # extract numeric midpoints
    mids = np.array([float(rx.match(c).group(1)) for c in cols], dtype=float)

    # arithmetic mean/std across time
    mean_vals = dnd[cols].mean(axis=0, skipna=True).to_numpy()
    sigma_vals = dnd[cols].std(axis=0, ddof=ddof, skipna=True).to_numpy()

    # non-NaN count per bin (how many data points used)
    n_vals = dnd[cols].count(axis=0).to_numpy(dtype=int)

    # sort by diameter midpoints
    order = np.argsort(mids)
    return mids[order], mean_vals[order], sigma_vals[order], label, n_vals[order]

def check_meta(label, df):
    meta = (df.attrs.get("bin_meta") or {})
    mids = meta.get("mid_nm", [])
    lo   = meta.get("lower_nm", [])
    up   = meta.get("upper_nm", [])
    dlog = meta.get("dlogD", [])
    print(f"{label:6s}  nbins={sum(bool(re.match(r'.*(_Bin\\d+|n[_-]?Dp_?\\d+)$', c)) for c in df.columns)}  "
          f"mid={len(mids)}  lower={len(lo)}  upper={len(up)}  dlogD={dlog}")

def check_common_grid(
    frames: Dict[str, pd.DataFrame],
    *,
    ref_key: Optional[str] = None,   # which instrument to use as the reference grid
    round_to: Optional[str] = None,  # e.g. "S" to round all times to whole seconds before comparing
) -> Tuple[bool, pd.DatetimeIndex, pd.DataFrame]:
    """
    Check whether all DataFrames share the *same* time index (common grid).
    """
    if not frames:
        raise ValueError("No frames provided.")

    # normalize: ensure tz-aware & sorted
    norm = {}
    for k, df in frames.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{k}: index must be DatetimeIndex.")
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx = idx.sort_values()
        if round_to:
            idx = idx.round(round_to)
        if idx.has_duplicates:
            idx = pd.DatetimeIndex(np.unique(idx.values))
        tmp = df.copy()
        tmp.index = idx
        norm[k] = tmp

    # pick reference
    if ref_key is None:
        ref_key = max(norm, key=lambda k: len(norm[k]))
    ref_idx = norm[ref_key].index

    # intersection grid
    common_idx = ref_idx
    for k, df in norm.items():
        common_idx = common_idx.intersection(df.index)

    # build report
    rows = []
    for k, df in norm.items():
        idx = df.index
        diffs = idx.to_series().diff().dropna()
        med_step = diffs.median() if not diffs.empty else pd.NaT
        equal_ref = idx.equals(ref_idx)
        missing = ref_idx.difference(idx)
        extra   = idx.difference(ref_idx)
        rows.append({
            "instrument": k,
            "n": len(idx),
            "t_min": idx.min(),
            "t_max": idx.max(),
            "median_step": med_step,
            "n_unique_steps": diffs.value_counts().size if not diffs.empty else 0,
            "equal_ref": equal_ref,
            "n_missing_vs_ref": len(missing),
            "n_extra_vs_ref": len(extra),
        })

    report = pd.DataFrame(rows).set_index("instrument").sort_index()
    all_match = bool(report["equal_ref"].all())
    return all_match, common_idx, report


def align_to_common_grid(
    frames: Dict[str, pd.DataFrame],
    *,
    grid: Optional[pd.DatetimeIndex] = None,  # if None, will compute per mode/freq
    mode: str = "intersection",               # "intersection" | "union" | "ref" | "span"
    ref_key: Optional[str] = None,            # used if mode="ref"
    round_to: Optional[str] = "S",            # normalize time before aligning
    freq: Optional[str] = None,               # e.g. "1S" to force a regular grid (used with mode="span" or to densify)
    interpolate: bool = True,                 # interpolate numeric columns only
    interp_method: str = "time",
    interp_limit: Optional[int] = None,
    interp_limit_direction: str = "both",
    non_numeric_fill: str = "none",           # "none" | "ffill" | "bfill"
    protect_flag_cols: bool = True,           # never interpolate columns that look like flags/QC
) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """
    Align (and optionally interpolate) frames onto a common time grid.
    """
    if not frames:
        raise ValueError("No frames provided.")

    # --- normalize indices ---
    norm = {}
    for k, df in frames.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{k}: index must be DatetimeIndex.")
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx = idx.sort_values()
        if round_to:
            idx = idx.round(round_to)
        if idx.has_duplicates:
            idx = pd.DatetimeIndex(np.unique(idx.values))
        tmp = df.copy()
        tmp.index = idx
        norm[k] = tmp

    # --- decide grid if not supplied ---
    if grid is None:
        if mode == "intersection":
            g = None
            for df in norm.values():
                g = df.index if g is None else g.intersection(df.index)
            grid = g if g is not None else pd.DatetimeIndex([], tz="UTC")

        elif mode == "union":
            g = None
            for df in norm.values():
                g = df.index if g is None else g.union(df.index)
            grid = g if g is not None else pd.DatetimeIndex([], tz="UTC")

        elif mode == "ref":
            if ref_key is None or ref_key not in norm:
                raise ValueError("mode='ref' requires a valid ref_key present in frames.")
            grid = norm[ref_key].index

        elif mode == "span":
            if freq is None:
                raise ValueError("mode='span' requires freq (e.g., '1S').")
            tmin = min(df.index.min() for df in norm.values())
            tmax = max(df.index.max() for df in norm.values())
            grid = pd.date_range(tmin, tmax, freq=freq, tz="UTC")
        else:
            raise ValueError("mode must be one of: 'intersection', 'union', 'ref', 'span'.")

    # If user also passed freq with a non-regular grid, densify onto that frequency
    if freq is not None and (mode in {"intersection", "union", "ref"} or grid is not None):
        if len(grid):
            grid = pd.date_range(grid.min(), grid.max(), freq=freq, tz="UTC")

    # --- reindex & interpolate ---
    aligned: Dict[str, pd.DataFrame] = {}
    for k, df in norm.items():
        out = df.reindex(grid)

        # Choose columns to interpolate (numeric), but avoid flag-like fields
        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        if protect_flag_cols and num_cols:
            flag_like = [c for c in num_cols if re.search(r"(flag|qc)", c, flags=re.I)]
            interp_cols = [c for c in num_cols if c not in flag_like]
        else:
            interp_cols = num_cols

        if interpolate and len(interp_cols):
            out[interp_cols] = (
                out[interp_cols]
                .interpolate(method=interp_method, limit=interp_limit, limit_direction=interp_limit_direction)
            )

        if non_numeric_fill in {"ffill", "bfill"}:
            out = getattr(out, non_numeric_fill)()

        aligned[k] = out

    return aligned, grid

def filter_by_spectra_presence(
    frames: Dict[str, pd.DataFrame],
    *,
    col_prefix: str = "dNdlogDp",
    min_instruments: Optional[int] = None,      # default ⇒ require all *spectral* instruments
    extra_masks: Optional[Dict[str, pd.Series]] = None,  # e.g., {"FIMS": fims_qc.ne(2)}
    treat_nonpositive_as_nan: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """
    Keep only timestamps where at least `min_instruments` *spectral* instruments have any
    finite size-bin value from get_spectra(..., long=False). Non-spectral frames (e.g., CPC)
    are ignored when computing the keep mask but are still filtered to the kept times.
    """
    if not frames:
        raise ValueError("No frames provided.")

    # Union time index across ALL frames (including CPC)
    union_idx = None
    for df in frames.values():
        union_idx = df.index if union_idx is None else union_idx.union(df.index)
    if union_idx is None:
        union_idx = pd.DatetimeIndex([], tz="UTC")

    # Determine which frames are spectral (have non-empty spectra)
    spectral_keys: List[str] = []
    spectra_cache: Dict[str, pd.DataFrame] = {}
    for k, df in frames.items():
        _, spec_df = get_spectra(df, col_prefix=col_prefix, long=False)
        spectra_cache[k] = spec_df
        if not spec_df.empty:
            spectral_keys.append(k)

    # Build has-data mask table ONLY over spectral instruments
    has_data: Dict[str, pd.Series] = {}
    for k in spectral_keys:
        spec_df = spectra_cache[k]
        A = spec_df.to_numpy(dtype=float)
        if treat_nonpositive_as_nan:
            A = np.where(A > 0, A, np.nan)
        m = pd.Series(~np.all(~np.isfinite(A), axis=1), index=spec_df.index)
        has_data[k] = m.reindex(union_idx, fill_value=False).astype(bool)

    mask_table = pd.DataFrame(has_data, index=union_idx).astype(bool)

    # AND any extra masks (applied only where the key exists in mask_table)
    if extra_masks:
        for k, extra in extra_masks.items():
            if k in mask_table:
                mask_table[k] = mask_table[k] & extra.reindex(union_idx).fillna(False).astype(bool)

    # Default threshold = number of *spectral* instruments
    if min_instruments is None:
        min_needed = len(spectral_keys)
    else:
        min_needed = min(int(min_instruments), len(spectral_keys))

    if len(spectral_keys) == 0:
        # No spectral instruments → keep everything (don’t let CPC erase data)
        keep_mask = pd.Series(True, index=union_idx)
    else:
        keep_mask = mask_table.sum(axis=1) >= min_needed

    kept_index = union_idx[keep_mask]
    filtered = {k: frames[k].loc[frames[k].index.intersection(kept_index)] for k in frames}
    return filtered, keep_mask

# --- inlet-flag helpers ---

def find_flag_column(df: pd.DataFrame) -> Optional[str]:
    cand = [c for c in df.columns if re.search(r'flag', c, re.I) or re.search(r'inlet', c, re.I)]
    return cand[0] if cand else None

def flag_segments(series: pd.Series):
    """
    Given a time-indexed flag series, return a list of (t0, t1, value) segments
    where the flag is constant over [t0, t1].

    Requires DatetimeIndex; raises if not.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return []

    s = s.astype(int).sort_index()

    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(
            f"flag_segments expects a DatetimeIndex, got {type(s.index)}"
        )

    idx = s.index          # keep as DatetimeIndex (pandas Timestamps)
    vals = s.values        # ints

    segments: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    seg_start = 0
    seg_val = vals[0]

    # split whenever the flag changes
    for i in range(1, len(vals)):
        if vals[i] != seg_val:
            t0 = idx[seg_start]
            t1 = idx[i - 1]
            segments.append((t0, t1, seg_val))
            seg_start = i
            seg_val = vals[i]

    # final segment
    t0 = idx[seg_start]
    t1 = idx[-1]
    segments.append((t0, t1, seg_val))

    return segments

def flag_fractions(segments):
    """Return {flag_value: fraction_of_total_duration}."""
    tot = sum((t1 - t0).total_seconds() for t0, t1, _ in segments)
    if tot <= 0:
        return {}
    acc: Dict[int, float] = {}
    for t0, t1, val in segments:
        acc[val] = acc.get(val, 0.0) + (t1 - t0).total_seconds()
    return {val: dur / tot for val, dur in acc.items()}