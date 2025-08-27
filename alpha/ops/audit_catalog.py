from __future__ import annotations

"""Catalog definitions for repo audit.

This module contains the lists of expected CLI commands, task file mappings,
configuration files, and tests.  The audit utility uses these structures to
check the current repository.
"""

# Expected CLI commands exposed by ``alpha.app.cli``
EXPECTED_CLI_COMMANDS = [
    "fetch-data",
    "run-pipeline",
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

# Mapping of clusters and task identifiers to expected source files
TASK_FILE_MAP = {
    "structure": {
        "S-01": ["alpha/structure/swings.py"],
        "S-02": ["alpha/structure/events.py"],
        "S-03": ["alpha/structure/quality.py"],
        "S-04": ["alpha/viz/structure.py"],
        "S-05": ["alpha/structure/trend.py"],
    },
    "liquidity": {
        "LQ-01": ["alpha/liquidity/asia.py"],
        "LQ-02": ["alpha/liquidity/eq.py"],
        "LQ-03": ["alpha/liquidity/sweep.py"],
    },
    "poi": {
        "POI-01": ["alpha/poi/ob.py"],
        "POI-02": ["alpha/viz/poi.py"],
    },
    "execution": {
        "EX-01": [
            "alpha/execution/engine.py",
            "alpha/execution/triggers.py",
            "alpha/execution/risk.py",
        ]
    },
    "backtests": {
        "EX-02": ["alpha/backtest/vbt_bridge.py", "alpha/backtest/metrics.py"],
        "EX-03": [
            "alpha/backtest/bt_strategy.py",
            "alpha/backtest/bt_broker.py",
            "alpha/backtest/bt_runner.py",
        ],
    },
    "report": {
        "RP-01": ["alpha/report/build_report.py", "alpha/report/templates/base.html"],
    },
    "data": {
        "DT-01": [
            "alpha/data/ingest.py",
            "alpha/data/normalize.py",
            "alpha/data/validators.py",
            "alpha/data/resample.py",
            "alpha/data/merge.py",
            "alpha/data/providers/base.py",
            "alpha/data/providers/csv_local.py",
            "alpha/app/cli.py",
        ],
        "DT-02": [
            "alpha/data/providers/oanda_provider.py",
            "alpha/data/providers/dukascopy_provider.py",
        ],
    },
    "ops": {
        "OPS-01": ["alpha/ops/runner.py", "alpha/ops/stages.py", "alpha/ops/registry.py"],
        # OPS-03 refers to the audit itself; keep expected for completeness
        "OPS-03": ["alpha/ops/audit.py", "alpha/ops/audit_catalog.py", "alpha/app/cli.py"],
    },
}

# Expected configuration files
EXPECTED_CONFIGS = [
    "alpha/config/structure.yml",
    "alpha/config/liquidity.yml",
    "alpha/config/poi.yml",
    "alpha/config/execution.yml",
    "alpha/config/backtest.yml",
    "alpha/config/viz.yml",
    "alpha/config/report.yml",
    "alpha/config/data.yml",
    "alpha/config/instruments.yml",
    "alpha/config/pipeline.yml",
]

# Suggested tests that should exist in the repository
EXPECTED_TESTS = [
    "tests/test_ops_pipeline.py",
    "tests/test_data_ingest.py",
    "tests/test_viz_poi.py",
    "tests/test_execution_engine.py",
    "tests/test_backtest_vbt.py",
    "tests/test_backtest_bt.py",
    "tests/test_report_html.py",
    "tests/test_provider_oanda.py",
    "tests/test_provider_duka.py",
]

# Critical CLI commands – if missing the audit should exit with non-zero status
CRITICAL_CLI = {"fetch-data", "run-pipeline"}

