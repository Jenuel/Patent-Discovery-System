"""Unit tests for retrieval fusion helpers (run from apps/api: python -m unittest discover tests)."""
from __future__ import annotations

import unittest

from app.services.retrieval.fusion import ScoredMatch, to_scored_matches


class ToScoredMatchesTests(unittest.TestCase):
    def test_converts_result_dicts(self):
        raw = [{"id": "US1", "score": 0.5, "metadata": {"patent_id": "US1"}}]
        matches = to_scored_matches(raw)
        self.assertEqual(matches, [ScoredMatch(id="US1", score=0.5, metadata={"patent_id": "US1"})])

    def test_missing_fields_get_defaults(self):
        matches = to_scored_matches([{}])
        self.assertEqual(matches, [ScoredMatch(id="", score=0.0, metadata={})])


if __name__ == "__main__":
    unittest.main()
