from __future__ import annotations

import pandas as pd


COLUMN_LABELS = {
    "run_date": "判定日",
    "screen_version": "スクリーナー版",
    "universe_offset": "ユニバース位置",
    "generated_at": "生成日時",
    "raw_rank": "元スコア順位",
    "entry_priority_score": "注目度スコア",
    "entry_rank": "注目順位",
    "ticker": "ティッカー",
    "name": "銘柄名",
    "score": "総合スコア",
    "volume_subscore": "出来高スコア",
    "trend_subscore": "トレンドスコア",
    "structure_subscore": "形状スコア",
    "liquidity_subscore": "流動性スコア",
    "penalty_subscore": "減点スコア",
    "close": "終値",
    "turnover_million": "売買代金(百万円)",
    "volume": "出来高",
    "prev_day_volume": "前日出来高",
    "prev_day_vol_gt_20d": "前日出来高20日平均超",
    "prev_day_vol_ratio_20": "前日出来高倍率(20日)",
    "volume_ratio": "出来高倍率",
    "vol_accel_3": "出来高加速3日",
    "close_position_pct": "引け位置(%)",
    "upper_shadow_pct": "上ヒゲ(%)",
    "lower_shadow_pct": "下ヒゲ(%)",
    "body_pct": "実体(%)",
    "near_breakout_5": "5日高値接近",
    "event_pre_earnings_like": "イベント前っぽい形",
    "core_signal": "コアシグナル",
    "smart_money_absorb": "吸収サイン",
    "inst_accumulation": "機関仕込み",
    "inst_accumulation_strong": "強い機関仕込み",
    "absorption_candle": "吸収ローソク",
    "absorption_candle_strong": "強い吸収ローソク",
    "is_overheated": "過熱",
    "prev_change_pct": "前日騰落率(%)",
    "change_5d_pct": "5日騰落率(%)",
    "ma5_gap_pct": "5日線乖離(%)",
    "day_range_pct": "当日値幅(%)",
    "resistance_gap_pct": "レジスタンス差(%)",
    "ma5_slope": "5日線傾き(%)",
    "ma10_slope": "10日線傾き(%)",
    "ma25_slope": "25日線傾き(%)",
    "entry_date": "エントリー日",
    "entry_open": "エントリー始値",
    "next_open_return_pct": "翌日寄りリターン(%)",
    "day1_high_return_pct": "翌日高値リターン(%)",
    "day1_low_drawdown_pct": "翌日安値DD(%)",
    "day1_close_return_pct": "翌日終値リターン(%)",
    "day3_close_return_pct": "3日後終値リターン(%)",
    "day5_close_return_pct": "5日後終値リターン(%)",
    "max_drawdown_5d_pct": "5日最大DD(%)",
    "max_runup_5d_pct": "5日最大上昇(%)",
}

LABEL_TO_COLUMN = {ja: en for en, ja in COLUMN_LABELS.items()}

WATCHLIST_COLUMN_ORDER = [
    "判定日",
    "注目順位",
    "注目度スコア",
    "ティッカー",
    "銘柄名",
    "過熱",
    "吸収ローソク",
    "機関仕込み",
    "5日高値接近",
    "総合スコア",
    "終値",
    "売買代金(百万円)",
    "出来高倍率",
    "出来高加速3日",
    "引け位置(%)",
    "前日騰落率(%)",
    "5日騰落率(%)",
    "レジスタンス差(%)",
    "出来高スコア",
    "トレンドスコア",
    "形状スコア",
    "減点スコア",
]

SCORED_EXTRA_ORDER = [
    "エントリー日",
    "エントリー始値",
    "翌日寄りリターン(%)",
    "翌日高値リターン(%)",
    "翌日安値DD(%)",
    "3日後終値リターン(%)",
    "5日後終値リターン(%)",
    "5日最大DD(%)",
    "5日最大上昇(%)",
]


def normalize_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: LABEL_TO_COLUMN[col] for col in df.columns if col in LABEL_TO_COLUMN}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _apply_order(df: pd.DataFrame, ordered_labels: list[str]) -> pd.DataFrame:
    ordered_existing = []
    seen = set()
    for col in ordered_labels:
        if col in df.columns and col not in seen:
            ordered_existing.append(col)
            seen.add(col)
    remaining = [col for col in df.columns if col not in ordered_existing]
    return df[ordered_existing + remaining]


def format_watchlist_output(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.rename(columns=COLUMN_LABELS)
    return _apply_order(display_df, WATCHLIST_COLUMN_ORDER)


def format_scored_output(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.rename(columns=COLUMN_LABELS)
    ordered = WATCHLIST_COLUMN_ORDER + [col for col in SCORED_EXTRA_ORDER if col not in WATCHLIST_COLUMN_ORDER]
    return _apply_order(display_df, ordered)
