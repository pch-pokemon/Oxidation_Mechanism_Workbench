"""Configurable, bounded sampling of distinct composition grid candidates."""

from decimal import Decimal, InvalidOperation
import math
import numpy as np
import pandas as pd
from app.core import FEATURES, ELEMENTS, RANGES

SCALE = 1_000_000
ENUM_LIMIT = 100_000
MAX_DRAWS = 1_000_000
CONDITIONS = {
    "solu_temp": 1200,
    "solu_time": 12,
    "middle_temp": 0,
    "middle_time": 0,
    "aging_temp": 900,
    "aging_time": 100,
    "Temperature": 900,
    "E_Time": 100,
}


def defaults():
    return {
        "elements": {
            e: {
                "min": RANGES.get(e, (0, 0))[0],
                "max": RANGES.get(e, (0, 0))[1],
                "step": 0.5,
            }
            for e in ELEMENTS
            if e != "Co"
        },
        "co": {"min": 40, "max": 80},
        "conditions": dict(CONDITIONS),
    }


def number(value, label):
    if (
        isinstance(value, bool)
        or value is None
        or not isinstance(value, (int, float, str))
    ):
        raise ValueError(f"{label}须为有限数值。")
    try:
        d = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"{label}须为有限数值。") from None
    if not d.is_finite():
        raise ValueError(f"{label}须为有限数值。")
    return d


def composition(value, label):
    d = number(value, label)
    if not 0 <= d <= 100:
        raise ValueError(f"{label}须在 0–100 at.% 内。")
    if d * SCALE != (d * SCALE).to_integral_value():
        raise ValueError(f"{label}最多支持 6 位小数。")
    return int(d * SCALE)


def integer(value, label, minimum, maximum):
    d = number(value, label)
    if d != d.to_integral_value() or not minimum <= d <= maximum:
        raise ValueError(f"{label}须为 {minimum}–{maximum} 之间的整数。")
    return int(d)


def normalize(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("候选生成参数须为 JSON 对象。")
    settings = {} if settings is None else settings
    unknown = set(settings) - {"elements", "co", "conditions"}
    if unknown:
        raise ValueError("未知候选生成参数：" + ", ".join(sorted(unknown)))
    cfg = defaults()
    grids = []
    for section in ("elements", "co", "conditions"):
        supplied = settings.get(section, {})
        if not isinstance(supplied, dict):
            raise ValueError(f"{section} 参数须为 JSON 对象。")
        extra = set(supplied) - set(cfg[section])
        if extra:
            raise ValueError(f"{section} 含未知字段：" + ", ".join(sorted(extra)))
        for key, value in supplied.items():
            if section == "elements":
                if not isinstance(value, dict) or set(value) - {"min", "max", "step"}:
                    raise ValueError(f"{key} 须提供 min、max、step 参数。")
                cfg[section][key].update(value)
            else:
                cfg[section][key] = value
    for e, spec in cfg["elements"].items():
        lo = composition(spec["min"], f"{e} 下限")
        hi = composition(spec["max"], f"{e} 上限")
        step = composition(spec["step"], f"{e} 步长")
        if lo > hi:
            raise ValueError(f"{e} 下限不能大于上限。")
        if step <= 0:
            raise ValueError(f"{e} 步长须大于 0；固定成分请设置上下限相同。")
        count = (hi - lo) // step + 1
        if count > 1_000_000:
            raise ValueError(f"{e} 网格点超过 100 万，请增大步长或缩小范围。")
        cfg["elements"][e] = {
            "min": lo / SCALE,
            "max": hi / SCALE,
            "step": step / SCALE,
        }
        grids.append((e, lo, step, count))
    clo = composition(cfg["co"]["min"], "Co 余量下限")
    chi = composition(cfg["co"]["max"], "Co 余量上限")
    if clo > chi:
        raise ValueError("Co 余量下限不能大于上限。")
    cfg["co"] = {"min": clo / SCALE, "max": chi / SCALE}
    for key, value in cfg["conditions"].items():
        d = number(value, key)
        if d < 0 or (key in ("Temperature", "E_Time") and d <= 0):
            raise ValueError("热处理温度/时间不得为负；氧化温度及暴露时间须大于 0。")
        f = float(d)
        if not math.isfinite(f):
            raise ValueError(f"{key}超出可表示数值范围。")
        cfg["conditions"][key] = f
    min_sum = sum(lo for _, lo, _, _ in grids)
    max_sum = sum(lo + step * (count - 1) for _, lo, step, count in grids)
    if min_sum > 100 * SCALE - clo or max_sum < 100 * SCALE - chi:
        raise ValueError(
            "所设成分网格与 Co 余量范围没有交集，无法生成候选。请调整成分范围或 Co 上下限。"
        )
    return cfg, grids, clo, chi


def generate_candidates(n=180, seed=42, settings=None):
    """Not a Cartesian sweep: sample without replacement, with explicit work limits."""
    n = integer(n, "候选数量", 2, 800)
    seed = integer(seed, "随机种子", 0, 2**32 - 1)
    cfg, grids, clo, chi = normalize(settings)
    rng = np.random.default_rng(seed)
    counts = [g[3] for g in grids]
    grid_size = math.prod(counts)
    if grid_size < n:
        raise ValueError(
            f"成分网格总共只有 {grid_size} 个不同组合，少于请求的 {n} 个候选。请降低生成数量、扩大范围或减小步长。"
        )
    lo = np.array([g[1] for g in grids], dtype=np.int64)
    step = np.array([g[2] for g in grids], dtype=np.int64)

    def co_values(indices):
        return 100 * SCALE - (lo + indices * step).sum(axis=1)

    selected = []
    eligible = None
    if grid_size <= ENUM_LIMIT:
        indices = np.column_stack(np.unravel_index(np.arange(grid_size), counts))
        co = co_values(indices)
        valid = indices[(co >= clo) & (co <= chi)]
        eligible = len(valid)
        if eligible < n:
            raise ValueError(
                f"满足 Co 余量范围的不同网格组合只有 {eligible} 个，少于请求的 {n} 个。请降低生成数量或调整成分/Co 范围。"
            )
        selected = valid[rng.choice(eligible, size=n, replace=False)].tolist()
        draws = grid_size
        strategy = "enumerate_then_sample_without_replacement"
    else:
        seen = set()
        draws = 0
        while len(selected) < n and draws < MAX_DRAWS:
            size = min(10_000, MAX_DRAWS - draws)
            indices = np.column_stack([rng.integers(c, size=size) for c in counts])
            co = co_values(indices)
            draws += size
            for idx in indices[(co >= clo) & (co <= chi)]:
                key = tuple(int(x) for x in idx)
                if key not in seen:
                    seen.add(key)
                    selected.append(key)
                    if len(selected) == n:
                        break
        if len(selected) < n:
            raise ValueError(
                f"已达到 {MAX_DRAWS:,} 次有界抽样上限，仅找到 {len(selected)} 个不同可行状态，未生成完整请求。请扩大 Co 范围、调整成分网格或降低候选数量；该提示不代表已穷举全部组合。"
            )
        strategy = "bounded_rejection_sample_without_replacement"
    rows = []
    for i, idx in enumerate(selected):
        amounts = lo + np.array(idx, dtype=np.int64) * step
        row = dict.fromkeys(FEATURES, 0.0)
        row.update({g[0]: int(a) / SCALE for g, a in zip(grids, amounts)})
        row["Co"] = int(100 * SCALE - amounts.sum()) / SCALE
        row.update(cfg["conditions"])
        row.update(
            sample_id=f"CAND{i+1:04d}",
            alloy_id=f"SYN{i+1:04d}",
            Ref_source_num=f"GENERATED-SRC{i%12+1:02d}",
            condition_id=f"{row['Temperature']:g}C_{row['E_Time']:g}h",
        )
        rows.append(row)
    info = {
        "source": "generated",
        "requested_count": n,
        "generated_count": len(rows),
        "seed": seed,
        "settings": cfg,
        "sampling": strategy,
        "unfiltered_grid_combinations": grid_size,
        "eligible_grid_combinations": eligible,
        "grid_draws": draws,
        "grid_points": {g[0]: g[3] for g in grids},
        "effective_grid_max": {e: (l + s * (c - 1)) / SCALE for e, l, s, c in grids},
        "notes": "非 Co 元素从 min 起按 step 取值且不超过 max；Co 为补足 100 at.% 的余量。随机抽取不同成分状态，非全网格扫描。来源编号为合成占位标识，不代表文献来源。",
    }
    return pd.DataFrame(rows), info
