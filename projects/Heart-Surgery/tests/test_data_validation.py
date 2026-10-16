import unittest

import pandas as pd

from data_validation import parse_year, prepare_data


BASE_ROW = {
    "Region": "Capital District",
    "Procedure": "All PCI",
    "Year of Hospital Discharge": 2015,
    "Number of Cases": 100,
    "Number of Deaths": 2,
    "Expected Mortality Rate": 1.5,
    "Observed Mortality Rate": 2.0,
    "Comparison Results": "Rate not different than Statewide Rate",
}


class DataValidationTests(unittest.TestCase):
    def test_parse_year_preserves_numeric_years_and_parses_dates(self):
        values = pd.Series([2015, "2016-07-01", "not-a-year"])
        parsed = parse_year(values)
        self.assertEqual(parsed.iloc[0], 2015)
        self.assertEqual(parsed.iloc[1], 2016)
        self.assertTrue(pd.isna(parsed.iloc[2]))

    def test_prepare_data_coerces_numeric_fields_and_drops_optional_identifiers(self):
        row = {**BASE_ROW, "Number of Cases": "100", "Facility ID": "facility-1"}
        result = prepare_data(pd.DataFrame([row]))
        self.assertNotIn("Facility ID", result)
        self.assertEqual(result["Number of Cases"].iloc[0], 100)

    def test_prepare_data_rejects_invalid_numeric_values(self):
        row = {**BASE_ROW, "Number of Deaths": "unknown"}
        with self.assertRaisesRegex(ValueError, "Number of Deaths"):
            prepare_data(pd.DataFrame([row]))

    def test_prepare_data_rejects_missing_required_columns(self):
        row = {key: value for key, value in BASE_ROW.items() if key != "Procedure"}
        with self.assertRaisesRegex(ValueError, "Procedure"):
            prepare_data(pd.DataFrame([row]))

    def test_prepare_data_rejects_out_of_range_rates(self):
        row = {**BASE_ROW, "Observed Mortality Rate": 101}
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            prepare_data(pd.DataFrame([row]))

    def test_prepare_data_rejects_impossible_counts(self):
        row = {**BASE_ROW, "Number of Deaths": 101}
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            prepare_data(pd.DataFrame([row]))


if __name__ == "__main__":
    unittest.main()
