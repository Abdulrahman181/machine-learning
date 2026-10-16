"""Regression tests for the dependency-free notebook contract validator."""
from __future__ import annotations

import unittest

import validate_notebook


class NotebookContractTests(unittest.TestCase):
    def test_static_contract_passes(self) -> None:
        self.assertEqual(validate_notebook.main(), 0)


if __name__ == "__main__":
    unittest.main()
