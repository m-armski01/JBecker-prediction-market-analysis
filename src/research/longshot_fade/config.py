"""Paths, split boundaries, constants and JSON config loading for the longshot-fade study."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[3]

# Source data (read-only)
TRADES_DIR = ROOT / "data" / "kalshi" / "trades"
MARKETS_DIR = ROOT / "data" / "kalshi" / "markets"

# Committed study files
RESEARCH_DIR = ROOT / "research" / "longshot_fade"
CONFIG_DIR = RESEARCH_DIR / "config"
RESULTS_DIR = RESEARCH_DIR / "results"
FIGURES_DIR = RESEARCH_DIR / "figures"
PROTOCOL_JSON = CONFIG_DIR / "protocol_v1.json"
FROZEN_JSON = CONFIG_DIR / "frozen.json"
PROTOCOL_MD = RESEARCH_DIR / "PROTOCOL.md"
DEVIATIONS_MD = RESEARCH_DIR / "DEVIATIONS.md"
HOLDOUT_LOG_MD = RESEARCH_DIR / "HOLDOUT_LOG.md"
PANEL_MANIFEST_JSON = RESULTS_DIR / "panel_manifest.json"

# Generated data (gitignored)
DATA_OUT = ROOT / "data" / "research" / "longshot_fade"
PANEL_DIR = DATA_OUT / "panel"
HOLDA_OUTCOMES_DIR = DATA_OUT / "holdA_outcomes"
BACKFILL_DIR = DATA_OUT / "backfill"
HOLDB_DIR = DATA_OUT / "holdB"
TMP_DIR = DATA_OUT / "tmp"

UTC = timezone.utc
SAMPLE_START = datetime(2024, 10, 1, tzinfo=UTC)
VAL_START = datetime(2025, 7, 1, tzinfo=UTC)
HOLDA_START = datetime(2025, 10, 1, tzinfo=UTC)

# (name, start inclusive, end exclusive or None for open-ended)
LOCAL_SPLITS: list[tuple[str, datetime, datetime | None]] = [
    ("train", SAMPLE_START, VAL_START),
    ("val", VAL_START, HOLDA_START),
    ("holdA", HOLDA_START, None),
]
SPLIT_RANK = {"train": 0, "val": 1, "holdA": 2, "holdB": 3}
OUTCOME_VISIBLE_SPLITS = frozenset({"train", "val"})
HOLDOUT_SPLITS = frozenset({"holdA", "holdB"})

FINANCE_GROUP = "Finance"
N_BASE = 100
SENSITIVITY_SIZES = (10, 1000)
SEED = 20251001
BOOT_ITERATIONS = 1000
BONFERRONI_M = 12
T_MIN = 2.0
CI_LO_MIN = 0.0
HOLD_DAYS_FLOOR = 1.0 / 24.0
TOP_K = 3
HOLDB_WAIT_DAYS = 7
HOLDB_MIN_EVENTS = 300

BRANCH = "prereg/longshot-fade"
TAG_PROTOCOL = "protocol-v1"
TAG_FROZEN = "prereg-frozen"
TAG_HOLDOUT_A = "holdout-a-run"
TAG_HOLDOUT_B = "holdout-b-run"


@dataclass(frozen=True)
class StrategyConfig:
    """One grid point: NO-cost band, category filter and liquidity floor."""

    band_lo: int
    band_hi: int
    category_filter: str
    liquidity_floor: float

    @property
    def id(self) -> str:
        return f"b{self.band_lo}-{self.band_hi}_{self.category_filter}_liq{int(self.liquidity_floor)}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategyConfig:
        return cls(
            band_lo=int(d["band_lo"]),
            band_hi=int(d["band_hi"]),
            category_filter=str(d["category_filter"]),
            liquidity_floor=float(d["liquidity_floor"]),
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, sort_keys=False) + "\n")


def load_protocol(path: Path = PROTOCOL_JSON) -> dict[str, Any]:
    """Load protocol_v1.json and check it agrees with the constants in this module."""
    proto = load_json(path)
    validate_protocol(proto)
    return proto


def validate_protocol(proto: dict[str, Any]) -> None:
    errors = []
    if proto.get("protocol_version") != 1:
        errors.append("protocol_version must be 1")
    grid_spec = proto["grid"]
    if grid_spec["n_configs"] != 12 or len(grid(proto)) != 12:
        errors.append("grid must have exactly 12 configurations")
    if grid_spec["finance_group_label"] != FINANCE_GROUP:
        errors.append("finance group label mismatch")
    if proto["order_size"]["base_contracts"] != N_BASE:
        errors.append("base order size mismatch")
    if tuple(proto["order_size"]["sensitivity_contracts"]) != SENSITIVITY_SIZES:
        errors.append("sensitivity sizes mismatch")
    boot = proto["stats"]["bootstrap"]
    if boot["seed"] != SEED or boot["iterations"] != BOOT_ITERATIONS:
        errors.append("bootstrap seed/iterations mismatch")
    if proto["stats"]["bonferroni_m"] != BONFERRONI_M:
        errors.append("bonferroni m mismatch")
    dec = proto["decision"]["pass_requires_all"]
    if dec["t_event_min"] != T_MIN or dec["boot_ci_lo_strictly_greater_than"] != CI_LO_MIN:
        errors.append("decision thresholds mismatch")
    hb = proto["splits"]["holdB"]
    if hb["outcome_fetch_min_days_after_pull"] != HOLDB_WAIT_DAYS or hb["power_gate_min_event_clusters"] != (
        HOLDB_MIN_EVENTS
    ):
        errors.append("holdB wait/power gate mismatch")
    for name, start, end in LOCAL_SPLITS:
        spec = proto["splits"][name]
        if _parse_ts(spec["start_utc"]) != start or _parse_ts(spec["end_utc_exclusive"]) != end:
            errors.append(f"split {name} boundary mismatch")
    if errors:
        raise ValueError("protocol_v1.json is inconsistent with config.py: " + "; ".join(errors))


def _parse_ts(val: str | None) -> datetime | None:
    if val is None:
        return None
    return datetime.fromisoformat(val.replace("Z", "+00:00"))


def grid(proto: dict[str, Any] | None = None) -> list[StrategyConfig]:
    """The 12 configurations in protocol order: band (outer), category filter, liquidity floor."""
    if proto is None:
        proto = load_json(PROTOCOL_JSON)
    g = proto["grid"]
    return [
        StrategyConfig(int(lo), int(hi), cat, float(floor))
        for lo, hi in g["bands_no_cost_cents"]
        for cat in g["category_filters"]
        for floor in g["liquidity_floors_usd"]
    ]


def quarter_label(ts: datetime) -> str:
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def quarter_windows(start: datetime, end: datetime) -> list[tuple[str, datetime, datetime]]:
    """Calendar quarters intersecting [start, end), clipped to that interval."""
    out = []
    y, q = start.year, (start.month - 1) // 3
    while True:
        qs = datetime(y, 3 * q + 1, 1, tzinfo=UTC)
        qe = datetime(y + (q == 3), (3 * (q + 1)) % 12 + 1, 1, tzinfo=UTC)
        if qs >= end:
            break
        lo, hi = max(qs, start), min(qe, end)
        if lo < hi:
            out.append((quarter_label(qs), lo, hi))
        y, q = (y + 1, 0) if q == 3 else (y, q + 1)
    return out


def connect(memory_limit: str = "8GB", temp_dir: Path | None = TMP_DIR) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute("SET enable_progress_bar=false")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir is not None:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{temp_dir}'")
    return con


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def git_head(cwd: Path = ROOT) -> str:
    return git("rev-parse", "HEAD", cwd=cwd)


def git_is_clean(cwd: Path = ROOT) -> bool:
    return git("status", "--porcelain", cwd=cwd) == ""


def utc_now() -> datetime:
    return datetime.now(UTC)


def rel(path: Path, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
