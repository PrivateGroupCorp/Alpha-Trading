"""Catalog of expected files and default templates for project doctor."""

from __future__ import annotations

REQUIRED_FILES = {
    "alpha/config/instruments.yml": """symbols:\n  EURUSD:\n    asset_class: \"fx\"\n    pip_size: 0.0001\n    tick_size: 0.00001\n    aliases:\n      oanda: \"EUR_USD\"\n      dukascopy: \"EURUSD\"\n      yfinance: \"EURUSD=X\"\n      ccxt: \"EUR/USDT\"\n""",
    "alpha/config/data.yml": """profiles:\n  default:\n    tz: \"UTC\"\n    provider: \"csv_local\"\n    chunk_days: 5\n    rate_limit_per_min: 60\n    backoff_policy: {base_s: 1.0, max_s: 60.0, factor: 2.0}\n    drop_incomplete_ohlc: true\n    fill_volume_nan_with_zero: true\n    save_raw: true\n    resample_to: [\"M15\",\"H1\"]\n    csv_local:\n      raw_glob: \"/mnt/data/EURUSD!_{TF}_*.csv\"\n    oanda:\n      account_type: \"practice\"\n      api_key_env: \"OANDA_API_KEY\"\n      host: \"https://api-fxpractice.oanda.com\"\n      price_source: \"mid\"\n      max_candles_per_call: 5000\n      granularity_map: {M1: \"M1\", M15: \"M15\", H1: \"H1\", D1: \"D\"}\n    dukascopy:\n      base_url: \"https://datafeed.dukascopy.com/datafeed\"\n      aggregate_to: \"M1\"\n      max_days_per_chunk: 2\n      parallel_days: 2\n""",
    "alpha/config/structure.yml": "profiles:\n  default: {}\n",
    "alpha/config/liquidity.yml": "profiles:\n  default: {}\n",
    "alpha/config/poi.yml": "profiles:\n  default: {}\n",
    "alpha/config/execution.yml": "profiles:\n  default: {}\n",
    "alpha/config/backtest.yml": "profiles:\n  default: {}\n",
    "alpha/config/viz.yml": "profiles:\n  default: {}\n",
    "alpha/config/report.yml": "profiles:\n  default: {}\n",
    "alpha/config/pipeline.yml": "profiles:\n  default: {}\n",
    "alpha/config/scheduler.yml": "schedules: {}\n",
    "alpha/config/qa.yml": "qa: {}\n",
    "alpha/report/templates/base.html": """<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\" />\n  <title>{{ title|default('Report') }}</title>\n</head>\n<body>\n  {% block content %}{% endblock %}\n</body>\n</html>\n""",
    "alpha/report/templates/partials/overview.html": "<div id=\"overview\">{% block overview %}{% endblock %}</div>\n",
    "alpha/report/templates/partials/section.html": "<section>{% block section %}{% endblock %}</section>\n",
    ".env.example": "# Keys & Webhooks\nOANDA_API_KEY=\nSLACK_WEBHOOK_URL=\nTG_BOT_TOKEN=\nTG_CHAT_ID=\nSMTP_URL=\n",
    "project/status/board.yml": "todo: []\n",
}

REQUIREMENTS = [
    "pandas",
    "numpy",
    "pyyaml",
    "pyarrow",
    "fastparquet",
    "matplotlib",
    "plotly",
    "vectorbt",
    "backtrader",
    "ccxt",
    "yfinance",
    "croniter",
    "jinja2",
    "python-dateutil",
    "requests",
]

REQUIRED_CLIS = [
    "fetch-data",
    "run-pipeline",
    "repo-audit",
    "schedule-run",
    "qa-run",
    "analyze-structure-swings",
    "analyze-structure-events",
    "analyze-structure-events-quality",
    "analyze-structure-viz",
    "analyze-structure-trend",
    "analyze-liquidity-asia",
    "analyze-liquidity-eq",
    "analyze-liquidity-sweep",
    "analyze-poi-ob",
    "analyze-poi-viz",
    "run-execution",
    "run-backtest-vbt",
    "run-backtest-bt",
    "generate-report",
]
