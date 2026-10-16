import ast
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "Black Friday Sales Prediction.ipynb"


class RepositoryStaticValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_json_and_code_parse(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])))

    def test_outputs_are_cleared(self):
        for cell in self.notebook["cells"]:
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))

    def test_portable_data_configuration(self):
        self.assertIn("BLACK_FRIDAY_DATA", self.code)
        self.assertNotRegex(self.code, r"[A-Za-z]:\\\\|/Users/|/home/|/mnt/")

    def test_leakage_aware_pipeline_and_metrics_present(self):
        for token in (
            "GroupShuffleSplit",
            "ColumnTransformer",
            "Pipeline",
            "OneHotEncoder",
            "mean_absolute_error",
            "mean_squared_error",
            "r2_score",
        ):
            self.assertIn(token, self.code)
        self.assertIn('features.pop("User_ID")', self.code)
        self.assertIn('features.drop(columns=["Product_ID"])', self.code)
        self.assertNotIn("LabelEncoder", self.code)

    def test_input_and_split_guards_present(self):
        for token in (
            "df.columns.is_unique",
            "df.empty",
            "Purchase contains missing values",
            "np.isfinite",
            "User_ID contains missing values",
            "At least two distinct User_ID groups",
            "X_train.empty or X_test.empty",
            "Unsupported feature dtypes",
            "RUN_CV=True requires at least three distinct User_ID groups",
        ):
            self.assertIn(token, self.code)

    def test_supporting_files_exist(self):
        for name in ("README.md", "requirements.txt", ".gitignore"):
            self.assertTrue((REPO / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
