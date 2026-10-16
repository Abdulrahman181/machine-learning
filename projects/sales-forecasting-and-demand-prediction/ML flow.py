"""CLI entry point for the reproducible forecasting pipeline.

Example: python "ML flow.py" --data /path/to/retail_store_inventory.csv
"""
from src.forecasting import main

if __name__ == "__main__":
    raise SystemExit(main())
