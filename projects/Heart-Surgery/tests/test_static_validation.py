import unittest
from pathlib import Path

from validate_notebook import NOTEBOOK, validate_notebook


class NotebookValidationTests(unittest.TestCase):
    def test_notebook_exists(self):
        self.assertTrue(Path(NOTEBOOK).is_file())

    def test_notebook_is_valid_without_data_or_ml_dependencies(self):
        self.assertEqual(validate_notebook(), [])


if __name__ == "__main__":
    unittest.main()
