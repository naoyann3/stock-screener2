from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
WATCHLISTS_DIR = RESULTS_DIR / "watchlists"
SCORED_DIR = RESULTS_DIR / "scored"
REPORTS_DIR = RESULTS_DIR / "reports"

SCREEN_VERSION = "v4"
DEFAULT_FORWARD_DAYS = (1, 3, 5)


def ensure_results_dirs():
    for path in (RESULTS_DIR, WATCHLISTS_DIR, SCORED_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
