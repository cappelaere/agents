"""
UN/LOCODE Lookup API (Final Clean Version)
-----------------------------------------
FastAPI service to retrieve UN/LOCODE given a port name (+ optional state, country).

Designed for CSVs with headers like the UNECE dumps:
  "Ch","LO","CODE","Name","NameWoDiacritics","SubDiv","Function","Status","Date","IATA","Coordinates","Remarks"

Key features
- Python 3.13 + Pydantic v2 compatible
- No hard dependency on pydantic-settings or RapidFuzz (both optional)
- Robust CSV reading (encoding & delimiter auto/fallback). Force via env if needed
- Header overrides via env (NAME_FIELD, STATE_FIELD, COUNTRY_FIELD, CODE_FIELD)
- Exact & fuzzy search (RapidFuzz if installed; difflib fallback)
- Safe JSON (NaN/NA sanitized) + helpful debug endpoints

Run
  pip install fastapi uvicorn pandas pydantic
  # optional for better fuzzy matching
  pip install rapidfuzz
  export CSV_PATH=/absolute/path/to/ports.csv
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Environment (optional)
  CSV_PATH        : path to CSV (default: ./ports.csv)
  CSV_ENCODING    : utf-8 | utf-8-sig | cp1252 | latin-1 ...
  CSV_SEP         : ',' | ';' | '	' (tab). If unset, auto-detect
  DEFAULT_LIMIT   : default list size (int, default 5)
  DEFAULT_SCORE   : default min similarity (0-100, default 80)
  NAME_FIELD      : override header for port name (e.g., "Name")
  STATE_FIELD     : override header for state/sub-division (e.g., "SubDiv")
  COUNTRY_FIELD   : override header for 2-letter country (e.g., "LO")
  CODE_FIELD      : override header for UN/LOCODE (e.g., "CODE")
"""
from __future__ import annotations

import os
import logging
from typing import List, Optional
import unicodedata

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------------
# Optional fuzzy backend
# ---------------------------------
try:  # RapidFuzz (preferred)
    from rapidfuzz import fuzz, process  # type: ignore
    _FUZZY_BACKEND = "rapidfuzz"
except Exception:  # Fallback to difflib
    from difflib import SequenceMatcher
    fuzz = process = None  # type: ignore
    _FUZZY_BACKEND = "difflib"

# ---------------------------------
# Logging
# ---------------------------------
logger = logging.getLogger("unlocode_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------
# Settings (env-based)
# ---------------------------------
CSV_PATH = os.getenv("CSV_PATH", "./ports.csv")
CSV_ENCODING = os.getenv("CSV_ENCODING")  # None -> auto
CSV_SEP = os.getenv("CSV_SEP")            # ',', ';', or '	' ; None -> auto
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "5"))
DEFAULT_SCORE = int(os.getenv("DEFAULT_SCORE", "80"))

# Header overrides
NAME_FIELD = os.getenv("NAME_FIELD")
STATE_FIELD = os.getenv("STATE_FIELD")
COUNTRY_FIELD = os.getenv("COUNTRY_FIELD")
CODE_FIELD = os.getenv("CODE_FIELD")

# ---------------------------------
# Models
# ---------------------------------
class QueryItem(BaseModel):
    name: str = Field(..., description="Port name to search (e.g., 'Port of Los Angeles')")
    state: Optional[str] = Field(None, description="State/region code or name (optional)")
    country: Optional[str] = Field(None, description="Country code or name (optional)")
    limit: Optional[int] = Field(None, ge=1, le=50, description="Max results to return")
    min_score: Optional[int] = Field(None, ge=0, le=100, description="Fuzzy match threshold")
    exact: bool = Field(False, description="Require exact normalized match when True")

    @field_validator("name")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("name is required")
        return v

class MatchResult(BaseModel):
    port_name: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    unlocode: Optional[str] = None
    score: Optional[int] = Field(None, description="Fuzzy similarity score (0-100)")

# ---------------------------------
# Helpers
# ---------------------------------
_df_cache: Optional[pd.DataFrame] = None

REQUIRED_CANON = ["port_name", "state", "country", "unlocode"]


def _normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = " ".join(s.split())
    return s.upper()


def _val_safe(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def df_to_records_safe(df: pd.DataFrame):
    return df.replace({pd.NA: None, np.nan: None}).to_dict(orient="records")


def _apply_header_overrides(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer explicit env mappings if present
    if NAME_FIELD and NAME_FIELD in df.columns:
        df = df.rename(columns={NAME_FIELD: "port_name"})
    elif "NameWoDiacritics" in df.columns:
        df = df.rename(columns={"NameWoDiacritics": "port_name"})
    elif "Name" in df.columns:
        df = df.rename(columns={"Name": "port_name"})

    if STATE_FIELD and STATE_FIELD in df.columns:
        df = df.rename(columns={STATE_FIELD: "state"})
    elif "SubDiv" in df.columns:
        df = df.rename(columns={"SubDiv": "state"})

    if COUNTRY_FIELD and COUNTRY_FIELD in df.columns:
        df = df.rename(columns={COUNTRY_FIELD: "country"})
    elif "LO" in df.columns:
        df = df.rename(columns={"LO": "country"})

    if CODE_FIELD and CODE_FIELD in df.columns:
        df = df.rename(columns={CODE_FIELD: "unlocode"})
    elif "CODE" in df.columns:
        df = df.rename(columns={"CODE": "unlocode"})

    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _apply_header_overrides(df)

    # Ensure canonical columns exist
    for c in REQUIRED_CANON:
        if c not in df.columns:
            df[c] = None

    # Auto-derive country from UN/LOCODE prefix if missing
    if df["country"].isna().all() or (df["country"].astype(str).str.len() == 0).all():
        df["country"] = df["unlocode"].astype(str).str.upper().str.slice(0, 2)

    # Clean canonical columns
    df["unlocode"] = df["unlocode"].astype(str).str.upper().str.replace(" ", "", regex=False)

    # Normalize empty strings/NaN → None for JSON safety
    for col in ["port_name", "state", "country", "unlocode"]:
        df[col] = df[col].replace({"": None})
        df[col] = df[col].where(df[col].notna(), None)

    # Add normalized helpers used for search
    df["norm_port_name"] = df["port_name"].map(_normalize_text)
    df["norm_state"] = df["state"].map(_normalize_text)
    df["norm_country"] = df["country"].map(_normalize_text)
    df["norm_join"] = df["norm_port_name"] + "|" + df["norm_state"] + "|" + df["norm_country"]

    keep = REQUIRED_CANON + ["norm_port_name", "norm_state", "norm_country", "norm_join"]
    return df[keep].copy()


def _read_csv_smart(path: str) -> pd.DataFrame:
    enc_candidates = [CSV_ENCODING] if CSV_ENCODING else []
    enc_candidates += ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    sep_candidates = [CSV_SEP] if CSV_SEP else [None, ",", ";", "	"]

    last_err: Optional[Exception] = None
    for enc in enc_candidates:
        if enc is None:
            continue
        for sep in sep_candidates:
            try:
                return pd.read_csv(path, encoding=enc, sep=sep, engine="python")
            except Exception as e:
                last_err = e
                continue

    # Last resort: salvage with replacement
    try:
        with open(path, "r", encoding="latin-1", errors="replace") as f:
            return pd.read_csv(f, sep=CSV_SEP if CSV_SEP else None, engine="python")
    except Exception as e:
        raise e if last_err is None else last_err


def load_dataframe(force: bool = False) -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None and not force:
        return _df_cache
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH}")

    raw = _read_csv_smart(CSV_PATH)
    df = _standardize_columns(raw)
    _df_cache = df
    logger.info("Loaded %d rows from %s (backend=%s)", len(df), CSV_PATH, _FUZZY_BACKEND)
    return df

# ---------------------------------
# Matching
# ---------------------------------

def exact_filter(df: pd.DataFrame, q: QueryItem) -> pd.DataFrame:
    mask = df["norm_port_name"] == _normalize_text(q.name)
    if q.state:
        mask &= df["norm_state"] == _normalize_text(q.state)
    if q.country:
        mask &= df["norm_country"] == _normalize_text(q.country)
    return df[mask]


def fuzzy_search(df: pd.DataFrame, q: QueryItem) -> List[MatchResult]:
    """Weighted fuzzy search. If country/state provided, operate on that subset only."""
    # Restrict by hard filters first if provided
    name_norm = _normalize_text(q.name)
    state_norm = _normalize_text(q.state)
    country_norm = _normalize_text(q.country)

    base = df
    if country_norm:
        base = base[base["norm_country"] == country_norm]
    if state_norm:
        base = base[base["norm_state"] == state_norm]

    limit = q.limit or DEFAULT_LIMIT
    min_score = q.min_score or DEFAULT_SCORE

    results: List[MatchResult] = []

    if _FUZZY_BACKEND == "rapidfuzz":
        from rapidfuzz import process, fuzz  # type: ignore
        # Candidate shortlist from name similarity
        name_matches = process.extract(
            name_norm, base["norm_port_name"].tolist(), scorer=fuzz.WRatio, limit=max(limit * 5, 50), score_cutoff=max(min_score - 20, 0)
        )
        if name_matches:
            idxs = [idx for _, _, idx in name_matches]
            sub = base.iloc[idxs].copy()
        else:
            sub = base

        def boost_row(row) -> int:
            n = process.extractOne(name_norm, [row["norm_port_name"]], scorer=fuzz.WRatio)[1]
            s = process.extractOne(state_norm, [row["norm_state"]], scorer=fuzz.WRatio)[1] if state_norm else 0
            c = process.extractOne(country_norm, [row["norm_country"]], scorer=fuzz.WRatio)[1] if country_norm else 0
            return int(0.75 * n + 0.15 * s + 0.10 * c)

        scored = [(i, boost_row(r)) for i, r in sub.iterrows()]
        scored = [t for t in scored if t[1] >= min_score]
        scored.sort(key=lambda t: t[1], reverse=True)
        for i, sc in scored[:limit]:
            row = sub.iloc[i]
            results.append(MatchResult(
                port_name=_val_safe(row["port_name"]),
                state=_val_safe(row["state"]),
                country=_val_safe(row["country"]),
                unlocode=_val_safe(row["unlocode"]),
                score=int(sc),
            ))
        return results

    # difflib fallback
    from difflib import SequenceMatcher
    def ratio(a: str, b: str) -> int:
        if not a or not b:
            return 0
        return int(SequenceMatcher(a=a, b=b).ratio() * 100)

    scores = []
    for i, r in base.iterrows():
        n = ratio(name_norm, r["norm_port_name"])  # primary
        s = ratio(state_norm, r["norm_state"]) if state_norm else 0
        c = ratio(country_norm, r["norm_country"]) if country_norm else 0
        sc = int(0.75 * n + 0.15 * s + 0.10 * c)
        if sc >= min_score:
            scores.append((i, sc))
    scores.sort(key=lambda t: t[1], reverse=True)

    for i, sc in scores[:limit]:
        row = base.iloc[i]
        results.append(MatchResult(
            port_name=_val_safe(row["port_name"]),
            state=_val_safe(row["state"]),
            country=_val_safe(row["country"]),
            unlocode=_val_safe(row["unlocode"]),
            score=int(sc),
        ))
    return results

    # difflib fallback
    from difflib import SequenceMatcher
    def ratio(a: str, b: str) -> int:
        if not a or not b:
            return 0
        return int(SequenceMatcher(a=a, b=b).ratio() * 100)

    scores = []
    for i, r in df.iterrows():
        n = ratio(name_norm, r["norm_port_name"])  # primary
        s = ratio(state_norm, r["norm_state"]) if state_norm else 0
        c = ratio(country_norm, r["norm_country"]) if country_norm else 0
        sc = int(0.7*n + 0.15*s + 0.15*c)
        if sc >= min_score:
            scores.append((i, sc))
    scores.sort(key=lambda t: t[1], reverse=True)

    for i, sc in scores[:limit]:
        row = df.iloc[i]
        results.append(MatchResult(
            port_name=_val_safe(row["port_name"]),
            state=_val_safe(row["state"]),
            country=_val_safe(row["country"]),
            unlocode=_val_safe(row["unlocode"]),
            score=int(sc),
        ))
    return results

# ---------------------------------
# FastAPI app
# ---------------------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        df = load_dataframe(force=True)
        logger.info("Lifespan startup: rows=%d, columns=%s", len(df), list(df.columns))
    except Exception as e:
        logger.exception("Startup failed: %s", e)
        # Let the app start; errors will be surfaced via /ports/health
    yield

# Create app before route decorators
app = FastAPI(title="UN/LOCODE Lookup API", version="3.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# -------------
# Error helpers
# -------------

def _error_payload(kind: str, message: str, details: dict | None = None):
    payload = {"error": kind, "message": message}
    if details is not None:
        payload["details"] = details
    return payload

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=_error_payload("UnprocessableEntity", "Validation failed", {"errors": exc.errors()}))

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # If detail is already a dict, pass through; else wrap it
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = _error_payload("HTTPException", str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=content)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content=_error_payload("ServerError", str(exc)))
# Lifespan handler replaces deprecated     df = load_dataframe(force=True)
    logger.info("Startup OK: rows=%d, columns=%s", len(df), list(df.columns))

@app.get("/ports/health", tags=["meta"]) 
def health():
    df = load_dataframe()
    return {
        "status": "ok",
        "rows": int(len(df)),
        "csv_path": CSV_PATH,
        "fuzzy_backend": _FUZZY_BACKEND,
        "default_limit": DEFAULT_LIMIT,
        "default_score": DEFAULT_SCORE,
    }

@app.get("/ports/columns", tags=["debug"]) 
def columns():
    df = load_dataframe()
    return {"columns": list(df.columns), "sample": df_to_records_safe(df.head(3))}

@app.get("/ports/peek", tags=["debug"]) 
def peek(n: int = 5):
    df = load_dataframe()
    n = max(1, min(int(n), 50))
    return df_to_records_safe(df.head(n))

@app.post("/ports/reload", tags=["debug"]) 
def reload_data():
    try:
        load_dataframe(force=True)
        return {"reloaded": True, "rows": int(len(load_dataframe()))}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=_error_payload("BadRequest", str(e)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=_error_payload("ServerError", str(e)))

@app.get("/ports/search", response_model=List[MatchResult], tags=["lookup"]) 
def search(
    name: str = Query(..., description="Port name"),
    state: Optional[str] = Query(None, description="State/region (optional)"),
    country: Optional[str] = Query(None, description="Country (optional)"),
    limit: Optional[int] = Query(None, ge=1, le=50),
    min_score: Optional[int] = Query(None, ge=0, le=100),
    exact: bool = Query(False, description="Exact normalized match only"),
):
    df = load_dataframe()
    q = QueryItem(name=name, state=state, country=country, limit=limit, min_score=min_score, exact=exact)

    # Hard filters first for exact path as well
    base = df
    if q.country:
        base = base[base["norm_country"] == _normalize_text(q.country)]
    if q.state:
        base = base[base["norm_state"] == _normalize_text(q.state)]

    if q.exact:
        hits = base[base["norm_port_name"] == _normalize_text(q.name)]
        results = [
            MatchResult(port_name=_val_safe(r.port_name), state=_val_safe(r.state), country=_val_safe(r.country), unlocode=_val_safe(r.unlocode), score=100)
            for r in hits.itertuples(index=False)
        ]
        return results[: (q.limit or DEFAULT_LIMIT)]

    # Lightweight substring match before fuzzy to catch simple cases
    name_norm = _normalize_text(q.name)
    mask_contains = base["norm_port_name"].str.contains(name_norm, na=False)
    prelim = base[mask_contains]
    prelim_results = [
        MatchResult(port_name=_val_safe(r.port_name), state=_val_safe(r.state), country=_val_safe(r.country), unlocode=_val_safe(r.unlocode), score=95)
        for r in prelim.itertuples(index=False)
    ]
    if prelim_results:
        return prelim_results[: (q.limit or DEFAULT_LIMIT)]

    # Fuzzy as final step
    # Build a temporary QueryItem limited to filters; fuzzy_search applies filters internally too
    return fuzzy_search(df, q)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8110, reload=True)
