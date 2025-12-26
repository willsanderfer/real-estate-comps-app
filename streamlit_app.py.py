# streamlit_app.py — v1.9.9 (garage = no jitter)
# - Robust file loader (CSV/TXT encoding+delimiter sniff)
# - Feature + Time analysis
# - Jitter disabled automatically when Garage Spaces is selected

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math
import os
from difflib import get_close_matches
import pydeck as pdk
from typing import Optional

import io, csv

st.set_page_config(page_title="Comparable Adjustment Explorer", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False, ttl=3600)
def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    # Excel straight-through
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    # Bytes for sniffing enc/delim
    raw = uploaded_file.getvalue()

    # 1) detect encoding by attempting small decode
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    chosen_enc = None
    sample_text = None
    for enc in encodings:
        try:
            sample_text = raw[:20000].decode(enc)  # strict
            chosen_enc = enc
            break
        except UnicodeDecodeError:
            continue
    if chosen_enc is None:
        # final fallback: lossy decode so the user still sees something
        chosen_enc = "latin-1"
        sample_text = raw[:20000].decode(chosen_enc, errors="replace")

    # 2) sniff delimiter (comma/tab/pipe/semicolon)
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t|;")
        sep = dialect.delimiter
    except Exception:
        # default to comma
        sep = ","

    # 3) read full file using detected encoding + delimiter
    # Use python engine for flexible parsing; skip bad lines instead of crashing
    text_io = io.StringIO(raw.decode(chosen_enc, errors="replace"))
    df = pd.read_csv(
        text_io,
        sep=sep,
        engine="python",
        on_bad_lines="skip"
    )
    return df


# ===================== CONSTANTS =====================
Y_COL_CANDIDATES = ["Sold Price", "Sale Price", "SoldPrice", "Close Price"]

FEATURE_SYNONYMS = {
    "SqFt Finished": [
        "SqFt -Total Finished","GLA","Square Feet","Living Area","Total Finished SqFt",
        "Finished SqFt","Sq Ft Finished","Gross Living Area","Above Grade Finished Area"
    ],
    "Above Grade Finished": [
        "Above Grade Finished","Above Grade Finished Area","AGLA","Above Grade Living Area",
        "GLA","Gross Living Area","Living Area","Square Feet","SqFt -Total Finished","Total Finished SqFt"
    ],
    "Basement SqFt Finished": [
        "Below Grade Finished","Below Grade Finished.","Basement SqFt Finished","Basement Finished SqFt",
        "Finished Basement SqFt","Basement Finished Area","Bsmt Fin SqFt","Basement Fin Sq Ft",
        "Basement Sq Ft Finished","Below Grade Finished Area","Below-Grade Finished Area",
        "BGFA","Finished Below Grade","Below Grade Finished, SqFt"
    ],
    "Basement Y/N": [
        "Basement Y/N","Basement Yes/No","Basement","Bsmt Y/N","Has Basement","Basement Present",
        "Basement Exists","BasementYn","Basement_YN","Basement?:","Basement?","Bsmt"
    ],
    "Garage Spaces": [
        "Garage Spaces","Garage","# Garage Spaces","Garage Y/N","Garage YN","Garage Spots","Garage Stalls"
    ],
}

# ===================== DATE/LAT/LON DETECTORS =====================
def find_first_date_col(df: pd.DataFrame) -> str | None:
    bad_name_snippets = ["list number","mls#","mls #","mls id","listing id","list no","record id","id"]
    for c in df.columns:
        name = str(c).lower().strip()
        if not any(k in name for k in ["date","close","sold","sale","contract","coe","closing","list date","listing date"]):
            continue
        if any(b in name for b in bad_name_snippets):
            continue
        s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        if s.notna().mean() < 0.5:
            continue
        yrs = s.dropna().dt.year
        if yrs.empty or yrs.min() < 1990 or yrs.max() > 2100:
            continue
        return c
    return None

def _is_lat_col(name: str) -> bool:
    n = str(name).lower()
    if "lot" in n:  # avoid false positives like "Lot Size"
        return False
    return ("lat" in n) or ("y coord" in n) or (n.strip() in {"y","latitude"})

def _is_lon_col(name: str) -> bool:
    n = str(name).lower()
    if "loan" in n:
        return False
    return ("lon" in n) or ("lng" in n) or ("long" in n) or ("x coord" in n) or (n.strip() in {"x","longitude"})

def find_lat_lon_cols(df: pd.DataFrame) -> tuple[str|None, str|None]:
    lat_col = lon_col = None
    for c in df.columns:
        if lat_col is None and _is_lat_col(c):
            lat_col = c
        if lon_col is None and _is_lon_col(c):
            lon_col = c
    return lat_col, lon_col

# ===================== SMALL HELPERS =====================
def _is_sqft_like_label(lbl: str) -> bool:
    l = (lbl or "").lower()
    return ("sqft" in l or "sq ft" in l or "square feet" in l or "gla" in l or "living area" in l or "finished" in l)

def unit_phrase_for_feature(feature_label: str, is_binary: bool) -> tuple[str|None, str]:
    if is_binary:
        return None, ""
    l = (feature_label or "").lower()
    if "garage" in l: return "per additional garage bay", "/bay"
    if "bed" in l:    return "per additional bedroom", "/bed"
    if "bath" in l:   return "per additional bathroom", "/bath"
    if "year" in l or "built" in l or "age" in l: return "per year", "/yr"
    if "acre" in l:   return "per acre", "/acre"
    if _is_sqft_like_label(l): return "per additional square foot", "/sqft"
    return f"per +1 {feature_label}", "/unit"

def clean_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        s = s.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def map_yes_no_to_binary(series: pd.Series) -> pd.Series:
    if series.dtype != "object":
        return series
    s = series.astype(str).str.strip().str.upper()
    yn = {"Y":1,"YES":1,"TRUE":1,"T":1,"1":1,"N":0,"NO":0,"FALSE":0,"F":0,"0":0}
    mapped = s.map(yn)
    out = series.copy()
    out.loc[mapped.notna()] = mapped.loc[mapped.notna()].astype(int)
    return out

def looks_discrete_integer(s: pd.Series, max_unique=12, tol=0.01) -> bool:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return False
    if s.nunique() <= max_unique:
        near_int = (s - s.round()).abs() <= tol
        return bool(near_int.all())
    return False

def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    for c in candidates:
        if c in cols: return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map: return lower_map[c.lower()]
    m = get_close_matches(candidates[0], cols, n=1, cutoff=0.6)
    return m[0] if m else None

def _tokens(s: str) -> set[str]:
    return set(str(s).lower().replace("_"," ").replace("-"," ").replace(".","").split())

def _viable_numeric(series: pd.Series, min_numeric_share=0.6) -> bool:
    s = clean_numeric(map_yes_no_to_binary(series))
    share = s.notna().mean() if len(s) else 0
    return share >= min_numeric_share and np.nanstd(s) > 0

def resolve_feature_column(df: pd.DataFrame, label: str) -> str | None:
    if label not in FEATURE_SYNONYMS:
        return None
    syns = FEATURE_SYNONYMS[label]
    cols = list(df.columns)

    required = {
        "SqFt Finished": {"sqft"} | {"gla","living","area","finished"},
        "Above Grade Finished": {"above","grade"} | {"gla","living","area","finished","sqft"},
        "Basement SqFt Finished": {"finished"} | {"fin"} | {"sqft"},
        "Basement Y/N": {"basement"} | {"yn","y/n","yes","no","present","exists","has"},
        "Garage Spaces": {"garage"} | {"space","spaces","stalls","spots"},
    }.get(label, set())

    if label == "Basement SqFt Finished":
        def _is_bg_finished(col: str) -> bool:
            t = _tokens(col)
            return (("basement" in t) or ({"below","grade"}.issubset(t))) and ({"finished","fin"}.intersection(t) or "sqft" in t)
        narrowed = [c for c in cols if _is_bg_finished(c)]
        if narrowed: cols = narrowed

    def _score(col: str) -> float:
        t = _tokens(col)
        score = 0.0
        for s in syns:
            ts = _tokens(s)
            score = max(score, len(t & ts) / max(1, len(ts)))
        for s in syns:
            if str(col).lower().startswith(str(s).lower()[:6]):
                score += 0.15
        return score
    
    # If Garage Spaces is blank, that means zero
    if "Garage Spaces" in df.columns:
        df["Garage Spaces"] = pd.to_numeric(df["Garage Spaces"], errors="coerce").fillna(0)


    for s in syns:
        if s in cols and _viable_numeric(df[s]): 
            return s
    lower_map = {c.lower(): c for c in cols}
    for s in syns:
        if s.lower() in lower_map and _viable_numeric(df[lower_map[s.lower()]]):
            return lower_map[s.lower()]

    cand = [c for c in cols if (not required) or required.issubset(_tokens(c))] if required else cols
    cand = sorted(cand, key=lambda c: _score(c), reverse=True)
    for c in cand:
        if _viable_numeric(df[c]):
            return c

    for s in syns:
        m = get_close_matches(s, cols, n=3, cutoff=0.82)
        for c in m:
            if (not required or required & _tokens(c)) and _viable_numeric(df[c]):
                return c
    return None

def regression_slope(x: np.ndarray, y: np.ndarray):
    if len(x) < 2 or np.nanstd(x) == 0:
        return np.nan, np.nan, np.nan
    m, b = np.polyfit(x, y, 1)
    r = pd.Series(x).corr(pd.Series(y))
    r2 = r*r if pd.notna(r) else np.nan
    return m, b, r2

def _is_gla_like(name: str) -> bool:
    n = name.lower()
    return (("gla" in n) or ("living" in n) or ("gross" in n) or ("total" in n) or ("above" in n and "grade" in n)) and ("basement" not in n and "below" not in n)

def compute_stats(df, y_col, x_col):
    x = df[x_col].values
    y = df[y_col].values
    m, _, r2 = regression_slope(x, y)
    median_ppsf = np.nan
    if _is_gla_like(x_col) or ("sqft" in x_col.lower() and "basement" not in x_col.lower() and "below" not in x_col.lower()):
        w = df[df[x_col] > 0].copy()
        median_ppsf = np.nan if w.empty else (w[y_col] / w[x_col]).median()
    return dict(slope=m, r2=r2, median_ppsf=median_ppsf, n=len(df))

def compute_binary_stats(df, y_col, x_col):
    present = sorted(df[x_col].dropna().astype(int).unique().tolist())
    has_both = (present == [0,1])
    res = {"n": len(df), "has_both": has_both}
    mean_no = df.loc[df[x_col]==0, y_col].mean() if 0 in present else np.nan
    mean_yes = df.loc[df[x_col]==1, y_col].mean() if 1 in present else np.nan
    res["mean_no"] = mean_no
    res["mean_yes"] = mean_yes
    if has_both:
        slope, _, r2 = regression_slope(df[x_col].values, df[y_col].values)
        res["slope"] = slope
        res["r2"] = r2
    else:
        res["slope"] = np.nan
        res["r2"] = np.nan
    return res

def fig_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def round_to_nearest_5_dollars(x: float) -> float:
    try:
        if x is None or np.isnan(x) or np.isinf(x):
            return np.nan
    except Exception:
        return np.nan
    return float(np.round(float(x) / 5.0) * 5.0)

# ======== TIME TOOLS ========
def _days_from_start(d: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(d, errors="coerce")
    base = dt.min()
    return (dt - base).dt.days.to_numpy()

def compute_time_adjustment(df: pd.DataFrame, y_col: str, date_col: str):
    d = pd.to_datetime(df[date_col], errors="coerce")
    msk = d.notna() & df[y_col].notna()
    w = df.loc[msk, [y_col, date_col]].copy()
    if w.empty or w[y_col].nunique() < 2:
        return dict(pct_per_month=np.nan, dollar_per_month=np.nan, r2=np.nan, n=0)

    t_days = _days_from_start(w[date_col])

    logp = np.log(w[y_col].values)
    if np.nanstd(t_days) > 0:
        m_pct, _, r2_pct = regression_slope(t_days, logp)
        pct_per_month = (np.exp(m_pct * 30.0) - 1.0) * 100.0
    else:
        pct_per_month, r2_pct = np.nan, np.nan

    if np.nanstd(t_days) > 0:
        m_dol, _, r2_dol = regression_slope(t_days, w[y_col].values)
        dollar_per_month = m_dol * 30.0
    else:
        dollar_per_month, r2_dol = np.nan, np.nan

    r2 = r2_pct if not np.isnan(r2_pct) else r2_dol
    return dict(pct_per_month=pct_per_month, dollar_per_month=dollar_per_month, r2=r2, n=len(w))

def make_time_scatter(df: pd.DataFrame, y_col: str, date_col: str, title: str):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    if df.empty:
        ax.set_title(title + " (no data)")
        return fig
    d = pd.to_datetime(df[date_col], errors="coerce")
    msk = d.notna() & df[y_col].notna()
    d = d[msk]; y = df.loc[msk, y_col].values
    t_days = _days_from_start(d)
    ax.scatter(d, y, s=32, label="Comps")
    if len(t_days) >= 2 and np.nanstd(t_days) > 0:
        m, b, _ = regression_slope(t_days, y)
        t_line = np.linspace(t_days.min(), t_days.max(), 200)
        y_line = m * t_line + b
        d0 = d.min()
        x_line = pd.to_datetime(d0) + pd.to_timedelta(t_line, unit="D")
        ax.plot(x_line, y_line, linewidth=2, label="Trend")
    ax.set_title(title); ax.set_xlabel(date_col); ax.set_ylabel(y_col)
    ax.grid(axis="y", linestyle="--", color="#e5e5e5", linewidth=0.8); ax.legend()
    fig.tight_LAYOUT()
    return fig

def key_takeaways_time(pct_per_month, dollar_per_month, r2, n):
    bullets = []
    if not np.isnan(pct_per_month):
        direction = "increasing" if pct_per_month >= 0 else "declining"
        bullets.append(f"Market trend is {direction} at {pct_per_month:.2f}% per month.")
    if not np.isnan(dollar_per_month):
        bullets.append(f"Equivalent dollar trend is ${dollar_per_month:,.2f} per month.")
    if not np.isnan(r2):
        strength = "weak" if r2 < 0.2 else "moderate" if r2 < 0.5 else "strong"
        bullets.append(f"Fit quality is {strength} (R²={r2:.2f}).")
    bullets.append(f"Sample size: n={n}.")
    return bullets[:5]

def plain_english_time(pct_per_month, dollar_per_month):
    if np.isnan(pct_per_month) and np.isnan(dollar_per_month):
        return "The model could not estimate a stable time trend for this data."
    parts = []
    if not np.isnan(pct_per_month):
        parts.append(f"{pct_per_month:.2f}% per month")
    if not np.isnan(dollar_per_month):
        parts.append(f"${dollar_per_month:,.2f} per month")
    if len(parts) == 2:
        return "Estimated market time adjustment is " + parts[0] + " (~" + parts[1] + ")."
    return "Estimated market time adjustment is " + parts[0]

# ===================== MAP BUILDERS =====================
def build_map_dataframe(original_df: pd.DataFrame, work_filtered_with_idx: pd.DataFrame,
                        lat_col: str, lon_col: str, y_col: str, x_col: str):
    coords = original_df.loc[:, [lat_col, lon_col]].copy()
    coords.columns = ["_lat", "_lon"]

    out = work_filtered_with_idx.merge(coords, left_on="index", right_index=True, how="left")

    out["_lat"] = pd.to_numeric(out["_lat"], errors="coerce")
    out["_lon"] = pd.to_numeric(out["_lon"], errors="coerce")
    out = out.dropna(subset=["_lat", "_lon"]).copy()

    out["lat"] = out["_lat"].astype(float)
    out["lon"] = out["_lon"].astype(float)
    out["price"] = pd.to_numeric(out[y_col], errors="coerce").astype(float)
    out["feature"] = pd.to_numeric(out[x_col], errors="coerce").astype(float)

    return out[["index", "lat", "lon", "price", "feature"]]

# ===================== FILTERS (no flagged logic) =====================
def filter_data(df_in: pd.DataFrame, y_col: str, x_col: str, date_col: str | None,
                price_rng, x_rng, date_rng, is_binary_x: bool) -> pd.DataFrame:
    out = df_in.copy()
    out = out[(out[y_col] >= price_rng[0]) & (out[y_col] <= price_rng[1])]
    if not is_binary_x:
        out = out[(out[x_col] >= x_rng[0]) & (out[x_col] <= x_rng[1])]
    if date_col and (date_col in out.columns) and date_rng is not None:
        d = pd.to_datetime(out[date_col], errors="coerce")
        mask = (d >= pd.to_datetime(date_rng[0])) & (d <= pd.to_datetime(date_rng[1]))
        out = out[mask]
    return out

# ===================== PLOTTING =====================
def make_scatter_figure(
    df, y_col, x_col, title,
    int_ticks=None, jitter_width=0.08, xtick_labels=None,
    removed_xy=None, disable_jitter: bool = False
):
    fig, ax = plt.subplots(figsize=(7.8,5.2))
    if df.empty:
        ax.set_title(title + " (no data)")
        return fig

    x_true = df[x_col].values
    y = df[y_col].values

    # jitter for discrete (but allow global disable, e.g., Garage Spaces)
    if int_ticks is not None and len(int_ticks) > 0:
        ax.set_xticks(int_ticks)
        if xtick_labels:
            ax.set_xticklabels(xtick_labels)
        else:
            ax.set_xticklabels([str(int(t)) for t in int_ticks])

        if disable_jitter:
            x_plot = x_true.astype(float)
        else:
            jw = float(jitter_width)
            # When removed markers are shown, keep diamonds exactly on the integer
            if removed_xy is not None:
                jw = 0.0
            rng = np.random.default_rng(42)
            x_plot = x_true.astype(float) + rng.uniform(-jw, jw, size=len(x_true))
    else:
        x_plot = x_true.astype(float)

    ax.scatter(x_plot, y, s=32, label="Comps")

    if removed_xy is not None:
        rx, ry = removed_xy
        # NOTE: no separate jitter on removed markers
        ax.scatter(rx, ry, facecolors='none', edgecolors='black', marker='D', s=90, label='Removed', zorder=3)

    if len(x_true) >= 2 and np.nanstd(x_true) > 0:
        m, b, _ = regression_slope(x_true, y)
        xline = np.linspace(np.nanmin(x_true), np.nanmax(x_true), 200)
        long_phrase, short_suffix = unit_phrase_for_feature(title.split(" — ")[-1], is_binary=False)
        ax.plot(xline, m*xline + b, linewidth=2, label=f"Fit: ${m:,.2f}{short_suffix}")

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(axis="y", linestyle="--", color="#e5e5e5", linewidth=0.8)
    fig.tight_layout()
    return fig

# ===================== GREEDY REMOVER (Admin) =====================
def greedy_remove_toward_target(df: pd.DataFrame, y_col: str, x_col: str, target_slope: float, max_removals: int):
    df = df.copy().reset_index(drop=False)
    df = df[['index', y_col, x_col]].dropna()
    if df.empty: return df, [], []
    cur_x = df[x_col].values; cur_y = df[y_col].values
    m, _, _ = regression_slope(cur_x, cur_y)
    removed_info = []
    direction = np.sign(target_slope - (m if not np.isnan(m) else 0))
    if direction == 0: return df, removed_info, []
    for step in range(max_removals):
        n = len(df)
        if n <= 2: break
        cur_x = df[x_col].values; cur_y = df[y_col].values
        m_cur, _, _ = regression_slope(cur_x, cur_y)
        if direction > 0 and m_cur >= target_slope - 1e-9: break
        if direction < 0 and m_cur <= target_slope + 1e-9: break
        best_idx = None; best_diff = None; best_new_m = None
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            x_try = cur_x[mask]; y_try = cur_y[mask]
            if len(x_try) < 2 or np.nanstd(x_try) == 0: continue
            m_try, _, _ = regression_slope(x_try, y_try)
            if direction > 0 and m_try < m_cur - 1e-9: continue
            if direction < 0 and m_try > m_cur + 1e-9: continue
            diff = abs(m_try - target_slope)
            if best_diff is None or diff < best_diff:
                best_diff = diff; best_idx = i; best_new_m = m_try
        if best_idx is None: break
        row = df.iloc[best_idx]
        removed_info.append({
            "orig_index": int(row['index']),
            y_col: float(row[y_col]),
            x_col: float(row[x_col]),
            "step": step + 1,
            "slope_after_removal": float(best_new_m)
        })
        df = df.drop(df.index[best_idx]).reset_index(drop=True)
    return df, removed_info, []

# ===================== AI NARRATIVE =====================
def infer_market_context(df: pd.DataFrame):
    geo_cols = ["City","Municipality","County","State","ST","Zip","ZIP","Postal Code","Neighborhood",
                "Subdivision","MLS Area","Area","Address","Street Address","Location","Region"]
    found = {}
    for c in df.columns:
        cl = str(c).strip()
        if cl in geo_cols or any(k in cl.lower() for k in ["city","county","state","zip","postal","neigh","subdiv","area","address","location","region"]):
            vc = df[cl].dropna().astype(str)
            if not vc.empty:
                found[cl] = vc.value_counts().head(3).index.tolist()
    city = state = county = zipc = None
    for k, v in found.items():
        lk = k.lower()
        if "city" in lk and v: city = v[0]
        if lk in ("state","st") and v: state = v[0]
        if "county" in lk and v: county = v[0].replace(" County","")
        if ("zip" in lk or "postal" in lk) and v: zipc = v[0]
    parts = []
    if city: parts.append(str(city))
    if state and state.upper() not in ("NAN","") and state not in parts: parts.append(str(state))
    if not parts and county: parts.append(f"{county} County")
    if not parts and zipc: parts.append(f"ZIP {zipc}")
    location_str = ", ".join(parts) if parts else None

    date_col = find_first_date_col(df)
    timeframe = None
    if date_col:
        d = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not d.empty:
            start = d.min().strftime("%b %Y"); end = d.max().strftime("%b %Y")
            timeframe = start if start == end else f"{start}–{end}"
    return {"location": location_str, "timeframe": timeframe}

def ai_summary_openai(feature_label, y_col, stats_before, stats_after, context):
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key)

    slope_use = stats_after.get("slope", np.nan)
    if np.isnan(slope_use):
        slope_use = stats_before.get("slope", np.nan)
    r2_use = stats_after.get("r2", np.nan)
    if np.isnan(r2_use):
        r2_use = stats_before.get("r2", np.nan)

    mppsf_after = stats_after.get("median_ppsf", np.nan)
    mppsf_before = stats_before.get("median_ppsf", np.nan)
    mppsf_val = mppsf_after if not np.isnan(mppsf_after) else mppsf_before
    mppsf_available = not np.isnan(mppsf_val)

    slope_rounded_5 = round_to_nearest_5_dollars(slope_use)

    lf = (feature_label or "").lower()
    is_binary_feature = ("y/n" in lf) or ("basement y" in lf)
    long_phrase, _ = unit_phrase_for_feature(feature_label, is_binary=is_binary_feature)
    coeff_phrase = None
    if not np.isnan(slope_rounded_5) and not is_binary_feature:
        coeff_phrase = f"${slope_rounded_5:,.2f} {long_phrase}"

    loc = context.get("location") or ""
    tf = context.get("timeframe") or ""
    where_when = ", ".join([p for p in [loc, tf] if p]).strip(", ")

    system_rules = (
        "You are a certified residential appraiser writing a professional adjustment explanation."
        " Use concise, neutral, 4–6 sentences. Mention regression of comparable sales in the subject’s"
        " competitive market area. Use the provided feature label verbatim. Note that atypical sales/outliers"
        " were removed. Conclude that the coefficient represents the market-supported contributory effect."
        " Use dollars with thousands separators. Do NOT mention internal tools or targets."
    )

    payload = {
        "market_context": where_when,
        "y_axis_label": y_col,
        "feature_label_exact": feature_label,
        "coefficient_literal_phrase": coeff_phrase,
        "r2_value": None if (r2_use is None or np.isnan(r2_use)) else float(r2_use),
        "median_price_per_sqft_available": bool(mppsf_available),
        "median_price_per_sqft": None if not mppsf_available else float(mppsf_val),
        "instructions": (
            "Use 'feature_label_exact' verbatim when naming the feature. "
            "If 'coefficient_literal_phrase' is provided, use it verbatim."
        ),
    }

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_rules},
            {"role": "user", "content": str(payload)},
        ],
    )
    return chat.choices[0].message.content.strip()

def ai_summary_fallback(feature_label, y_col, stats_before, stats_after, context):
    slope_use = stats_after.get("slope", np.nan)
    if np.isnan(slope_use):
        slope_use = stats_before.get("slope", np.nan)
    r2_use = stats_after.get("r2", np.nan)
    if np.isnan(r2_use):
        r2_use = stats_before.get("r2", np.nan)

    mppsf_after = stats_after.get("median_ppsf", np.nan)
    mppsf_before = stats_before.get("median_ppsf", np.nan)
    mppsf_val = mppsf_after if not np.isnan(mppsf_after) else mppsf_before

    slope_rounded_5 = round_to_nearest_5_dollars(slope_use)
    lf = (feature_label or "").lower()
    is_binary_feature = ("y/n" in lf) or ("basement y" in lf)
    long_phrase, _ = unit_phrase_for_feature(feature_label, is_binary=is_binary_feature)
    coeff_phrase = None
    if not np.isnan(slope_rounded_5) and not is_binary_feature:
        coeff_phrase = f"${slope_rounded_5:,.2f} {long_phrase}"

    loc = context.get("location")
    tf = context.get("timeframe")
    where_when = ", ".join([p for p in [loc, tf] if p]) if (loc or tf) else None

    lines = []
    lead = "Regression analysis of comparable sales"
    if where_when:
        lead += f" in {where_when}"
    lead += " was performed to estimate the contributory effect of the selected feature."
    lines.append(lead)

    if coeff_phrase:
        lines.append(f"The resulting coefficient indicates a typical sale price change of {coeff_phrase}.")
    else:
        lines.append("The resulting coefficient reflects the typical sale price difference attributable to the feature.")

    if not np.isnan(r2_use):
        lines.append(f"Model fit was quantified using R²={r2_use:.3f}, based on the filtered data set.")
    else:
        lines.append("Model fit was assessed on the filtered data set.")

    if not np.isnan(mppsf_val):
        lines.append(f"For reference, the median price per square foot was ${mppsf_val:,.2f} among usable observations.")

    lines.append("Atypical sales and statistical outliers were removed to reflect typical market behavior.")
    lines.append("These results provide a market-supported basis for the applied adjustment.")
    return " ".join(lines)

def ai_summary_always(feature_label, y_col, stats_before, stats_after, context):
    try:
        return ai_summary_openai(feature_label, y_col, stats_before, stats_after, context)
    except Exception:
        return ai_summary_fallback(feature_label, y_col, stats_before, stats_after, context)

# ===================== UI HEADER =====================
st.title("Comparable Adjustment Explorer")
st.caption("For appraisers. Upload MLS exports, pick a feature, view graph-first results with clear stats.")

uploaded = st.file_uploader("Upload MLS export (CSV/TXT/XLSX)", type=["csv","txt","xlsx","xls"])
if uploaded is None:
    with st.expander("How to use (3 steps)", expanded=False):
        st.markdown(
            "1) Upload a CSV/XLSX/TXT with a Sold Price column.\n"
            "2) Choose a comparison feature (SqFt, Basement, Garage, etc.).\n"
            "3) (Optional) Adjust to a target and review results."
        )
    st.stop()

try:
    df = load_table(uploaded)
except Exception as e:
    st.error("Couldn't read the file. Try re-exporting as CSV (comma-delimited) or Excel.")
    st.exception(e)
    st.stop()

df.columns = [c.strip() for c in df.columns]

# Y locked
y_col = pick_column(df, Y_COL_CANDIDATES)
if not y_col:
    st.error("Could not find a Sold Price column. Try renaming to one of: " + ", ".join(Y_COL_CANDIDATES))
    st.stop()
st.success(f"Y-axis locked to: {y_col}")

# Sidebar page switch (feature vs time)
mode = st.sidebar.radio("Analysis", ["Feature adjustment","Time adjustment"], index=0)

# ===================== TIME ADJUSTMENT PAGE =====================
if mode == "Time adjustment":
    date_col = find_first_date_col(df)
    if not date_col:
        st.error("Could not detect a date column (e.g., Close/Sold/Contract Date)."); st.stop()

    time_df = df[[y_col, date_col]].copy()
    time_df[y_col] = clean_numeric(time_df[y_col])
    time_df[date_col] = pd.to_datetime(time_df[date_col], errors="coerce")
    time_df = time_df.dropna(subset=[y_col, date_col]).reset_index(drop=True)
    if time_df.empty:
        st.error("No usable rows after cleaning."); st.stop()

    price_min, price_max = float(time_df[y_col].min()), float(time_df[y_col].max())
    dmin = time_df[date_col].min().date()
    dmax = time_df[date_col].max().date()

    with st.expander("Filters", expanded=True):
        c1, c2 = st.columns([1,1])
        with c1:
            pr = st.slider("Sold Price range", min_value=price_min, max_value=price_max,
                           value=(price_min, price_max), step=max(1.0, (price_max-price_min)/200.0))
        with c2:
            dr = st.date_input(f"{date_col} window", value=(dmin, dmax), min_value=dmin, max_value=dmax)

    f = time_df[(time_df[y_col].between(pr[0], pr[1])) &
                (time_df[date_col].between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1])))].copy()
    if f.empty:
        st.error("No rows match the current filters."); st.stop()

    stats_time = compute_time_adjustment(f, y_col, date_col)

    left_time, right_time = st.columns([2,1], gap="large")
    with left_time:
        fig_time = make_time_scatter(
            f,
            y_col, date_col,
            f"Sold Price over Time — {date_col} (filtered)"
        )
        st.pyplot(fig_time)

    with right_time:
        st.subheader("Current stats")
        st.metric("% per month", f"{stats_time['pct_per_month']:.2f}%" if not np.isnan(stats_time['pct_per_month']) else "—")
        st.metric("$/month", f"${stats_time['dollar_per_month']:,.2f}" if not np.isnan(stats_time['dollar_per_month']) else "—")
        st.metric("R²", f"{stats_time['r2']:.3f}" if not np.isnan(stats_time['r2']) else "—")
        st.caption(f"Comps: {stats_time['n']}")

        st.subheader("Downloads")
        st.download_button("Time chart PNG", fig_bytes(fig_time), file_name="time_adjustment.png")
        st.download_button("Filtered CSV (date + price)",
                           f[[date_col, y_col]].to_csv(index=False).encode(),
                           file_name="time_filtered.csv")
        st.download_button("Summary CSV",
                           pd.DataFrame([{
                               "date_column": date_col,
                               "pct_per_month": None if np.isnan(stats_time['pct_per_month']) else round(stats_time['pct_per_month'], 3),
                               "dollar_per_month": None if np.isnan(stats_time['dollar_per_month']) else round(stats_time['dollar_per_month']),
                               "R2": None if np.isnan(stats_time['r2']) else round(stats_time['r2'], 3),
                               "n": int(stats_time['n']),
                           }]).to_csv(index=False).encode(),
                           file_name="time_summary.csv")

    st.subheader("Key takeaways")
    for b in key_takeaways_time(stats_time["pct_per_month"], stats_time["dollar_per_month"], stats_time["r2"], stats_time["n"]):
        st.write("• " + b)
    st.subheader("Plain-English")
    st.write(plain_english_time(stats_time["pct_per_month"], stats_time["dollar_per_month"]))
    st.stop()

# ===================== FEATURE ADJUSTMENT PAGE =====================

# Feature choice
feature_label = st.selectbox("Compare Sold Price against:", list(FEATURE_SYNONYMS.keys()), index=0)
x_col = resolve_feature_column(df, feature_label)
if not x_col:
    st.error(f"Could not map “{feature_label}” to any column in your file.\nTry one similar to: {FEATURE_SYNONYMS[feature_label]}")
    st.stop()

# Clean + prep
work = df[[y_col, x_col]].copy()
work[y_col] = clean_numeric(work[y_col])
work[x_col] = clean_numeric(map_yes_no_to_binary(work[x_col]))

# If feature is Garage Spaces — treat blanks as 0
if "garage" in x_col.lower():
    work[x_col] = work[x_col].fillna(0)

# Drop rows that still have no price
work = work.dropna(subset=[y_col]).reset_index(drop=False)

if work.empty:
    st.error("No usable data after cleaning."); st.stop()

# Binary/discrete detection
uniq = np.sort(work[x_col].dropna().unique())
is_binary = len(uniq) <= 2 and set(np.unique(uniq).astype(int)) <= {0,1}
intish = looks_discrete_integer(work[x_col]) if not is_binary else True
if is_binary:
    int_ticks = [0,1]; xtick_labels = ["No","Yes"]
else:
    int_ticks = np.sort(work[x_col].round().astype(int).unique()) if intish else None
    xtick_labels = None

# Date range & base filters
date_col = find_first_date_col(df)
work_for_filters = work.copy()
global_date_min = global_date_max = None
if date_col:
    all_dates = pd.to_datetime(df[date_col], errors="coerce")
    work_for_filters[date_col] = all_dates.iloc[work_for_filters['index']].values
    if all_dates.notna().any():
        global_date_min = all_dates.min()
        global_date_max = all_dates.max()

price_min, price_max = float(work_for_filters[y_col].min()), float(work_for_filters[y_col].max())
x_min, x_max = float(work_for_filters[x_col].min()), float(work_for_filters[x_col].max())

with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        price_rng = st.slider(
            "Sold Price range", min_value=price_min, max_value=price_max,
            value=(price_min, price_max), step=max(1.0, (price_max-price_min)/200.0)
        )
    with c2:
        x_rng = st.slider(
            f"{x_col} range", min_value=x_min, max_value=x_max,
            value=(x_min, x_max), step=(1.0 if is_binary else max(1.0, (x_max-x_min)/200.0)),
            disabled=is_binary
        )
    with c3:
        if date_col and global_date_min is not None and global_date_max is not None:
            date_rng = st.date_input(
                f"{date_col} window",
                value=(global_date_min.date(), global_date_max.date()),
                min_value=global_date_min.date(), max_value=global_date_max.date()
            )
        else:
            date_rng = None

# Apply filters
work_filt = filter_data(
    df_in=work_for_filters, y_col=y_col, x_col=x_col, date_col=date_col,
    price_rng=price_rng, x_rng=(x_min, x_max) if is_binary else x_rng,
    date_rng=date_rng, is_binary_x=is_binary
)
if work_filt.empty:
    st.error("No rows match the current filters."); st.stop()

# Determine if this feature should have jitter disabled (Garage Spaces)
no_jitter_for_this_feature = ("garage" in feature_label.lower()) or ("garage" in x_col.lower())

# Map (collapsed by default)
with st.expander("Map of filtered comps", expanded=False):
    lat_col, lon_col = find_lat_lon_cols(df)
    if not lat_col or not lon_col:
        st.info("No latitude/longitude columns detected. Add 'Latitude' and 'Longitude' to enable the map.")
    else:
        try:
            map_df = build_map_dataframe(
                original_df=df,
                work_filtered_with_idx=work_filt[["index", x_col, y_col]].copy(),
                lat_col=lat_col, lon_col=lon_col,
                y_col=y_col, x_col=x_col
            )
            if map_df.empty:
                st.info("No mappable rows in the current filters.")
            else:
                price_min_m = float(map_df["price"].min())
                price_max_m = float(map_df["price"].max())
                radius = np.interp(map_df["price"], [price_min_m, price_max_m], [40, 140]).astype(int)

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df.assign(radius=radius),
                    get_position='[lon, lat]',
                    get_radius="radius",
                    get_fill_color=[52, 136, 189, 160],
                    pickable=True,
                )

                view_state = pdk.ViewState(
                    latitude=float(map_df["lat"].mean()),
                    longitude=float(map_df["lon"].mean()),
                    zoom=13.5, pitch=0
                )

                tooltip = {
                    "html": "<b>Price:</b> ${price}<br/><b>{x_name}:</b> {feature}",
                    "style": {"backgroundColor": "white", "color": "black"}
                }
                tooltip["html"] = tooltip["html"].replace("{x_name}", x_col)

                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
                st.caption("Sample of mapped comps")
                st.dataframe(
                    map_df.rename(columns={"price":"Sold Price", "feature": x_col})
                          .drop(columns=["index"])
                          .head(25)
                )
        except Exception as e:
            st.warning(f"Map could not be rendered: {e}")

# Stats (no flagged exclusions)
work_used = work_filt.copy()

# Baseline stats (for right-side metrics)
if is_binary:
    bin_stats = compute_binary_stats(work_used, y_col, x_col)
    default_target = 0.0 if not bin_stats["has_both"] or np.isnan(bin_stats["slope"]) else float(round(bin_stats["slope"]))
else:
    stats0 = compute_stats(work_used, y_col, x_col)
    m0, _, _ = regression_slope(work_used[x_col].values, work_used[y_col].values)
    default_target = 0.0 if np.isnan(m0) else float(round(m0))

# ===================== ADMIN TOOLS (kept) =====================
left_admin, right_admin = st.columns([2, 1], gap="large")
with left_admin:
    target = st.number_input(f"Target adjustment (price per +1 {feature_label})", value=default_target, step=1.0)
    n_total = len(work_used)
    default_max = max(1, min(math.floor(n_total * 0.25), 200))
    max_removals = st.slider("Max removals allowed", min_value=0, max_value=max(0, min(200, n_total - 2)), value=default_max)

with right_admin:
    st.subheader(f"Current stats — {feature_label}")
    if is_binary:
        if bin_stats["has_both"]:
            st.metric("Avg difference (Yes − No)", f"${bin_stats['slope']:,.2f}")
            st.metric("R²", f"{bin_stats['r2']:.3f}" if not np.isnan(bin_stats['r2']) else "—")
        else:
            st.metric("Avg difference (Yes − No)", "—")
            st.caption("Need both Yes and No to compute a difference.")
        st.caption(f"Comps: {bin_stats['n']}")
    else:
        st.metric("Price per +1", f"${stats0['slope']:,.2f}" if not np.isnan(stats0['slope']) else "—")
        st.metric("R²", f"{stats0['r2']:.3f}" if not np.isnan(stats0['r2']) else "—")
        if not np.isnan(stats0["median_ppsf"]):
            st.metric("Median $/sq ft", f"${stats0['median_ppsf']:,.2f}")
        st.caption(f"Comps: {stats0['n']}")

st.markdown("---")

# ===================== MAIN ACTION =====================
if st.button("Adjust to Target", type="primary"):
    kept_df, removed_info, _ = greedy_remove_toward_target(
        work_used[['index', y_col, x_col]].copy(), y_col, x_col, target, max_removals
    )
    kept_plot = kept_df[[x_col, y_col]].copy()

    # Overlays for removed
    removed_df = pd.DataFrame(removed_info)
    _rx = removed_df.get(x_col, pd.Series([], dtype=float)).values if not removed_df.empty else []
    _ry = removed_df.get(y_col, pd.Series([], dtype=float)).values if not removed_df.empty else []

    c1, c2 = st.columns([2, 1], gap="large")

    # --- Original (FILTERED) ---
    with c1:
        orig_fig = make_scatter_figure(
            work_filt[[x_col, y_col]], y_col, x_col,
            f"Original comps — {feature_label} (filtered; shows removals if applied)",
            int_ticks=( [0,1] if is_binary else (np.sort(work_filt[x_col].round().astype(int).unique()) if looks_discrete_integer(work_filt[x_col]) else None) ),
            xtick_labels=(["No","Yes"] if is_binary else None),
            removed_xy=(_rx, _ry) if len(_rx) > 0 else None,
            disable_jitter=no_jitter_for_this_feature
        )
        st.pyplot(orig_fig)
    st.caption("**Legend:** • Blue dot = Kept comps  • Black ◊ = Removed (to reach target)")
    with c2:
        st.subheader("Original stats")
        if is_binary:
            bs_all = compute_binary_stats(work_filt, y_col, x_col)
            if bs_all["has_both"]:
                st.metric("Avg difference (Yes − No)", f"${bs_all['slope']:,.2f}")
                st.metric("R²", f"{bs_all['r2']:.3f}" if not np.isnan(bs_all['r2']) else "—")
                st.caption(f"Means — No: {('$'+format(bs_all['mean_no'],',.2f')) if not np.isnan(bs_all['mean_no']) else '—'}   |   Yes: {('$'+format(bs_all['mean_yes'],',.2f')) if not np.isnan(bs_all['mean_yes']) else '—'}")
            else:
                st.metric("Avg difference (Yes − No)", "—")
                st.caption("Need both Yes and No to compute a difference.")
            st.caption(f"Comps: {bs_all['n']}")
        else:
            s0 = compute_stats(work_filt, y_col, x_col)
            st.metric("Price per +1", f"${s0['slope']:,.2f}" if not np.isnan(s0['slope']) else "—")
            st.metric("R²", f"{s0['r2']:.3f}" if not np.isnan(s0['r2']) else "—")
            if not np.isnan(s0["median_ppsf"]):
                st.metric("Median $/sq ft", f"${s0['median_ppsf']:,.2f}")
            st.caption(f"Comps: {s0['n']}")

    st.divider()

    # --- Adjusted (FINAL) ---
    with c1:
        final_fig = make_scatter_figure(
            kept_plot, y_col, x_col, f"Adjusted comps — {feature_label}",
            int_ticks=([0,1] if is_binary else (
                np.sort(kept_plot[x_col].round().astype(int).unique())
                if looks_discrete_integer(kept_plot[x_col]) else None)),
            xtick_labels=(["No","Yes"] if is_binary else None),
            removed_xy=None,  # no removed markers on the final chart
            disable_jitter=no_jitter_for_this_feature
        )
        st.pyplot(final_fig)
    st.caption("**Legend:** • Blue dot = Kept comps  • Line = Fit")
    with c2:
        st.subheader("Final stats")
        if is_binary:
            bs_final = compute_binary_stats(kept_plot, y_col, x_col)
            table_rows = {
                "Comps kept": [len(kept_plot)],
                "Removed": [len(removed_info)],
                "Avg diff (Yes−No)": [f"${bs_final['slope']:,.2f}" if bs_final["has_both"] and not np.isnan(bs_final["slope"]) else "—"],
                "R²": [f"{bs_final['r2']:.3f}" if bs_final["has_both"] and not np.isnan(bs_final["r2"]) else "—"],
            }
        else:
            sA = compute_stats(kept_plot, y_col, x_col)
            table_rows = {
                "Comps kept": [len(kept_plot)],
                "Removed": [len(removed_info)],
                "Price per +1": [f"${sA['slope']:,.2f}" if not np.isnan(sA['slope']) else "—"],
                "R²": [f"{sA['r2']:.3f}" if not np.isnan(sA['r2']) else "—"],
            }
            if not np.isnan(sA["median_ppsf"]):
                table_rows["Median $/sq ft"] = [f"${sA['median_ppsf']:,.2f}"]
        st.table(pd.DataFrame(table_rows))

        # ------- Downloads -------
        kept_csv = kept_df[[x_col, y_col]].to_csv(index=False).encode()
        removed_csv = removed_df.to_csv(index=False).encode()

        if is_binary:
            bs_all = compute_binary_stats(work_filt, y_col, x_col)
            bs_final = compute_binary_stats(kept_plot, y_col, x_col)
            summary_df = pd.DataFrame([{
                "feature_label": feature_label,
                "mapped_feature_column": x_col,
                "original_diff_yes_minus_no": None if not bs_all["has_both"] or np.isnan(bs_all["slope"]) else float(bs_all["slope"]),
                "original_r2": None if not bs_all["has_both"] or np.isnan(bs_all["r2"]) else float(bs_all["r2"]),
                "target_slope": float(target),
                "final_diff_yes_minus_no": None if not bs_final["has_both"] or np.isnan(bs_final["slope"]) else float(bs_final["slope"]),
                "final_r2": None if not bs_final["has_both"] or np.isnan(bs_final["r2"]) else float(bs_final["r2"]),
                "removed_count": len(removed_info),
            }])
        else:
            s0 = compute_stats(work_filt, y_col, x_col)
            sA = compute_stats(kept_plot, y_col, x_col)
            summary_df = pd.DataFrame([{
                "feature_label": feature_label,
                "mapped_feature_column": x_col,
                "original_slope": None if np.isnan(s0["slope"]) else float(s0["slope"]),
                "original_r2": None if np.isnan(s0["r2"]) else float(s0["r2"]),
                "target_slope": float(target),
                "final_slope": None if np.isnan(sA["slope"]) else float(sA["slope"]),
                "final_r2": None if np.isnan(sA["r2"]) else float(sA["r2"]),
                "original_median_ppsf": None if np.isnan(s0["median_ppsf"]) else float(s0["median_ppsf"]),
                "final_median_ppsf": None if np.isnan(sA["median_ppsf"]) else float(sA["median_ppsf"]),
                "removed_count": len(removed_info),
            }])

        st.subheader("Downloads")
        st.download_button("Kept comps CSV", kept_csv, file_name="kept_comps.csv")
        st.download_button("Removed comps CSV", removed_csv, file_name="removed_rows.csv")
        st.download_button("Summary CSV", summary_df.to_csv(index=False).encode(), file_name="adjustment_summary.csv")
        st.download_button("Original chart PNG", fig_bytes(orig_fig), file_name="original.png")
        st.download_button("Adjusted (final) PNG", fig_bytes(final_fig), file_name="adjusted_final.png")

    # ---- Market narrative ----
    st.divider()
    st.subheader("Market narrative")

    def _md_safe(s: str) -> str:
        return s.replace("$", r"\$").replace("_", r"\_")

    def _pretty_feature_name(label: str) -> str:
        m = {
            "SqFt Finished": "Gross Living Area",
            "Above Grade Finished": "Gross Living Area",
            "Basement SqFt Finished": "Basement Finish",
            "Basement Y/N": "Basement",
            "Garage Spaces": "Garage Spaces",
        }
        return m.get(label, label)

    def _unit_phrase(label: str, is_binary: bool) -> str:
        l = label.lower()
        if is_binary:
            # binary reads as yes/no difference
            return " (Yes vs No)"
        if "garage" in l: return " per additional garage bay"
        if "bed" in l:    return " per additional bedroom"
        if "bath" in l:   return " per additional bathroom"
        if "year" in l or "built" in l: return " per year"
        if "acre" in l:   return " per acre"
        if "sq" in l or "gla" in l or "finished" in l or "living area" in l:
            return " per additional square foot"
        return f" per +1 {label}"

    def build_appraiser_narrative(
    feature_label: str,
    slope: Optional[float],
    median_ppsf: Optional[float],
    context: dict,
    is_binary: bool
) -> str:

        feature_name = _pretty_feature_name(feature_label)
        where_when = ", ".join([p for p in [context.get("location"), context.get("timeframe")] if p])

        header = f"**{feature_name} Adjustment Commentary (Regression-Based):**"
        intro = (
            f"The {feature_name.lower()} adjustment was developed using regression analysis applied "
            "to a data set of comparable properties from within the subject’s competitive market area. "
            f"The analysis isolated the contributory effect of {feature_name.lower()} on sale price while controlling for "
            "other key variables such as location, condition, and amenities."
        )
        methods = (
            "Prior to model calibration, the data set was screened for accuracy, and statistical outliers or sales "
            "exhibiting atypical motivation or condition were removed to ensure a reliable representation of market behavior."
        )
        body = (
            "The resulting coefficient reflects the market-supported rate of change in sale price attributable to differences "
            f"in {feature_name.lower()} and provides a credible, data-driven basis for the applied adjustment."
        )

        lines = [
            header,
            "",
            intro,
            methods,
            body,
            ""
        ]

        if slope is not None and not np.isnan(slope):
            lines.append(f"**Indicated rate of change:** ${slope:,.2f}{_unit_phrase(feature_label, is_binary)}.")
        if (not is_binary) and (median_ppsf is not None) and (not np.isnan(median_ppsf)):
            lines.append(f"**Reference median $/sq ft:** ${median_ppsf:,.2f}.")

        return "\n\n".join(lines)




    # stats for narrative
    if is_binary:
        bs_all = compute_binary_stats(work_filt, y_col, x_col)
        bs_final = compute_binary_stats(kept_plot if 'kept_plot' in locals() else work_filt, y_col, x_col)
        slope_use = bs_final.get("slope", np.nan) if bs_final.get("has_both") else bs_all.get("slope", np.nan)
        median_use = np.nan  # not applicable for binary
    else:
        s_before = compute_stats(work_filt, y_col, x_col)
        s_after  = compute_stats(kept_plot if 'kept_plot' in locals() else work_filt, y_col, x_col)
        slope_use = s_after.get("slope", np.nan) if not np.isnan(s_after.get("slope", np.nan)) else s_before.get("slope", np.nan)
        # if it's a sqft-like feature, s_* already includes median_ppsf; prefer the "after" one when available
        median_use = s_after.get("median_ppsf", np.nan)
        if np.isnan(median_use):
            median_use = s_before.get("median_ppsf", np.nan)

    context_info = infer_market_context(df)
    narrative_text = build_appraiser_narrative(feature_label, slope_use, median_use, context_info, is_binary)
    st.markdown(_md_safe(narrative_text))

else:
    # No adjustment yet — show one baseline chart
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        base_fig = make_scatter_figure(
            work_filt[[x_col, y_col]], y_col, x_col,
            f"Comps — {feature_label} (filtered)",
            int_ticks=( [0,1] if is_binary else (np.sort(work_filt[x_col].round().astype(int).unique()) if looks_discrete_integer(work_filt[x_col]) else None) ),
            xtick_labels=(["No","Yes"] if is_binary else None),
            removed_xy=None,
            disable_jitter=no_jitter_for_this_feature
        )
        st.pyplot(base_fig)
    with c2:
        st.subheader("Current stats (filtered)")
        if is_binary:
            bs = compute_binary_stats(work_filt, y_col, x_col)
            if bs["has_both"]:
                st.metric("Avg difference (Yes − No)", f"${bs['slope']:,.2f}")
                st.metric("R²", f"{bs['r2']:.3f}" if not np.isnan(bs['r2']) else "—")
                st.caption(f"Means — No: {('$'+format(bs['mean_no'],',.2f')) if not np.isnan(bs['mean_no']) else '—'}   |   Yes: {('$'+format(bs['mean_yes'],',.2f')) if not np.isnan(bs['mean_yes']) else '—'}")
            else:
                st.metric("Avg difference (Yes − No)", "—")
                st.caption("Need both Yes and No to compute a difference.")
            st.caption(f"Comps: {bs['n']}")
        else:
            s = compute_stats(work_filt, y_col, x_col)
            st.metric("Price per +1", f"${s['slope']:,.2f}" if not np.isnan(s['slope']) else "—")
            st.metric("R²", f"{s['r2']:.3f}" if not np.isnan(s['r2']) else "—")
            if not np.isnan(s["median_ppsf"]):
                st.metric("Median $/sq ft", f"${s["median_ppsf"]:,.2f}")
            st.caption(f"Comps: {s['n']}")

    st.subheader("Downloads")
    st.download_button("Baseline chart PNG", fig_bytes(base_fig), file_name="comps_baseline.png")
    st.download_button("Filtered CSV (X,Y only)",
                       work_filt[[x_col, y_col]].to_csv(index=False).encode(),
                       file_name="filtered_xy.csv")

    st.info("Tip: Set a target and click **Adjust to Target** to see which comps would be removed to achieve that slope.")
