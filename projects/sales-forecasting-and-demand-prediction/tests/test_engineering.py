import unittest
from pathlib import Path

import pandas as pd

from src.forecasting import load_dataset, prepare_features


class EngineeringTests(unittest.TestCase):
    def test_missing_dataset_fails_with_actionable_message(self):
        with self.assertRaisesRegex(FileNotFoundError, "Dataset not found"):
            load_dataset(Path("/definitely/missing/sales.csv"))

    def test_targets_and_derived_target_proxies_are_not_features(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=2),
                "Units Sold": [1, 2],
                "Demand Forecast": [2, 3],
                "Inventory Level": [10, 11],
                "Inventory_Demand": [20, 33],
                "UnitsSold_Price": [5, 10],
            }
        )
        features = prepare_features(frame)
        self.assertNotIn("Units Sold", features.columns)
        self.assertNotIn("Demand Forecast", features.columns)
        self.assertNotIn("Inventory_Demand", features.columns)
        self.assertNotIn("UnitsSold_Price", features.columns)
        self.assertIn("Inventory Level", features.columns)


if __name__ == "__main__":
    unittest.main()
