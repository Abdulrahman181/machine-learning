import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_notebook import NOTEBOOK, validate_all_notebooks, validate_notebook


class NotebookStaticValidationTest(unittest.TestCase):
    def test_notebook_exists(self):
        self.assertTrue(Path(NOTEBOOK).is_file())

    def test_all_notebooks_are_portable_and_syntax_valid(self):
        self.assertEqual(validate_all_notebooks(), [])

    def test_validator_rejects_saved_output_and_execution_count(self):
        notebook = {
            "nbformat": 4,
            "cells": [
                {
                    "cell_type": "code",
                    "id": "cell-1",
                    "source": ["print('x')\n"],
                    "outputs": [{"output_type": "stream", "text": ["x\n"]}],
                    "execution_count": 1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            errors = validate_notebook(path)
        self.assertIn("cell 0 contains saved outputs", errors)
        self.assertIn("cell 0 contains an execution count", errors)


if __name__ == "__main__":
    unittest.main()
