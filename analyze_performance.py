from __future__ import annotations

from datetime import datetime

import pandas as pd

from config import REPORTS_DIR, SCORED_DIR, ensure_results_dirs


RETURN_COLUMNS = [
    "next_open_return_pct",
    "day1_high_return_pct",
    "day1_low_drawdown_pct",
    "day3_close_return_pct",
    "day5_close_return_pct",
    "max_drawdown_5d_pct",
    "max_runup_5d_pct",
]

FLAG_COLUMNS = [
    "near_breakout_5",
    "event_pre_earnings_like",
    "core_signal",
    "inst_accumulation",
    "inst_accumulation_strong",
    "absorption_candle",
    "absorption_candle_strong",
    "is_overheated",
]


def load_scored_data() -> pd.DataFrame:
    files = sorted(SCORED_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No scored CSV found in results/scored.")
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "sample_count": len(df),
        "win_rate_day3_pct": (df["day3_close_return_pct"] > 0).mean() * 100,
        "win_rate_day5_pct": (df["day5_close_return_pct"] > 0).mean() * 100,
        "avg_day1_high_return_pct": df["day1_high_return_pct"].mean(),
        "avg_day3_close_return_pct": df["day3_close_return_pct"].mean(),
        "avg_day5_close_return_pct": df["day5_close_return_pct"].mean(),
        "avg_max_drawdown_5d_pct": df["max_drawdown_5d_pct"].mean(),
        "avg_max_runup_5d_pct": df["max_runup_5d_pct"].mean(),
    }
    return pd.DataFrame([summary]).round(4)


def build_flag_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FLAG_COLUMNS:
        if col not in df.columns:
            continue
        subset = df[df[col] == 1].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "flag": col,
                "sample_count": len(subset),
                "win_rate_day3_pct": round((subset["day3_close_return_pct"] > 0).mean() * 100, 4),
                "avg_day1_high_return_pct": round(subset["day1_high_return_pct"].mean(), 4),
                "avg_day3_close_return_pct": round(subset["day3_close_return_pct"].mean(), 4),
                "avg_day5_close_return_pct": round(subset["day5_close_return_pct"].mean(), 4),
                "avg_max_drawdown_5d_pct": round(subset["max_drawdown_5d_pct"].mean(), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_day3_close_return_pct", "sample_count"], ascending=False)


def main():
    ensure_results_dirs()
    df = load_scored_data()
    df = df.dropna(subset=["day3_close_return_pct", "day5_close_return_pct"]).copy()
    if df.empty:
        print("Scored data exists, but no rows have enough forward bars yet.")
        return

    summary_df = build_summary(df)
    flag_df = build_flag_report(df)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = REPORTS_DIR / f"summary_{stamp}.csv"
    flags_path = REPORTS_DIR / f"flags_{stamp}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    flag_df.to_csv(flags_path, index=False, encoding="utf-8-sig")

    print("=== Overall Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    if not flag_df.empty:
        print("\n=== Flag Breakdown ===")
        print(flag_df.to_string(index=False))
        print(f"\nSaved: {flags_path}")


if __name__ == "__main__":
    main()
