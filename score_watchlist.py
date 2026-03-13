from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DEFAULT_FORWARD_DAYS, SCORED_DIR, WATCHLISTS_DIR, ensure_results_dirs


LOOKBACK_BUFFER_DAYS = 10
LOOKAHEAD_BUFFER_DAYS = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Score watchlist candidates with forward returns.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a watchlist CSV. Defaults to the latest file in results/watchlists.",
    )
    return parser.parse_args()


def latest_watchlist_path() -> Path:
    files = sorted(WATCHLISTS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No watchlist CSV found in results/watchlists.")
    return files[-1]


def fetch_history(ticker: str, run_date: pd.Timestamp) -> pd.DataFrame | None:
    start = (run_date - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)).date().isoformat()
    end = (run_date + pd.Timedelta(days=LOOKAHEAD_BUFFER_DAYS)).date().isoformat()
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=True,
            group_by="column",
        )
    except Exception as exc:
        print(f"fetch_history error: {ticker} {exc}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()].copy()
    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in need_cols):
        return None

    df = df[need_cols].dropna().copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df


def _pct(base: float, value: float) -> float:
    return round((value - base) / base * 100, 6)


def score_row(row: pd.Series) -> dict:
    run_date = pd.Timestamp(row["run_date"]).normalize()
    hist = fetch_history(row["ticker"], run_date)
    if hist is None:
        return {
            "entry_date": None,
            "entry_open": None,
            "next_open_return_pct": None,
            "day1_high_return_pct": None,
            "day1_low_drawdown_pct": None,
            "day3_close_return_pct": None,
            "day5_close_return_pct": None,
            "max_drawdown_5d_pct": None,
            "max_runup_5d_pct": None,
        }

    future = hist.loc[hist.index > run_date].copy()
    if future.empty:
        return {
            "entry_date": None,
            "entry_open": None,
            "next_open_return_pct": None,
            "day1_high_return_pct": None,
            "day1_low_drawdown_pct": None,
            "day3_close_return_pct": None,
            "day5_close_return_pct": None,
            "max_drawdown_5d_pct": None,
            "max_runup_5d_pct": None,
        }

    entry_bar = future.iloc[0]
    entry_open = float(entry_bar["Open"])
    result = {
        "entry_date": future.index[0].date().isoformat(),
        "entry_open": round(entry_open, 6),
        "next_open_return_pct": _pct(float(row["close"]), entry_open),
        "day1_high_return_pct": _pct(entry_open, float(entry_bar["High"])),
        "day1_low_drawdown_pct": _pct(entry_open, float(entry_bar["Low"])),
    }

    for days in DEFAULT_FORWARD_DAYS:
        if len(future) >= days:
            target_bar = future.iloc[days - 1]
            result[f"day{days}_close_return_pct"] = _pct(entry_open, float(target_bar["Close"]))
        else:
            result[f"day{days}_close_return_pct"] = None

    future_5d = future.head(5)
    if not future_5d.empty:
        result["max_drawdown_5d_pct"] = _pct(entry_open, float(future_5d["Low"].min()))
        result["max_runup_5d_pct"] = _pct(entry_open, float(future_5d["High"].max()))
    else:
        result["max_drawdown_5d_pct"] = None
        result["max_runup_5d_pct"] = None

    return result


def main():
    ensure_results_dirs()
    args = parse_args()
    input_path = args.input if args.input else latest_watchlist_path()
    df = pd.read_csv(input_path)
    if df.empty:
        raise SystemExit("Watchlist is empty.")

    scored_rows = []
    for _, row in df.iterrows():
        scored = score_row(row)
        merged = row.to_dict()
        merged.update(scored)
        scored_rows.append(merged)

    scored_df = pd.DataFrame(scored_rows)
    output_path = SCORED_DIR / input_path.name
    scored_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Scored CSV saved: {output_path}")


if __name__ == "__main__":
    main()
