"""Offline checks for the packaged corpus, index, and edition filter."""
import json
import unittest
from pathlib import Path

from rag_fiscal.candidate_retrieve import CandidateRetriever
from rag_fiscal.common import ROOT


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = CandidateRetriever(backend="chroma")

    def test_four_editions_and_source_bytes(self):
        self.assertEqual(set(self.retriever.editions), {2023, 2024, 2025, 2026})
        self.assertEqual(set(map(int, self.retriever.summary["by_edition"])),
                         set(self.retriever.editions))
        # Construction already verifies the extracted corpus and each source PDF hash.
        self.assertEqual(len(self.retriever.documents),
                         sum(self.retriever.summary["by_edition"].values()))

    def test_bm25_keeps_the_requested_edition(self):
        for year in self.retriever.editions:
            with self.subTest(year=year):
                result = self.retriever.search("Article premier impôt sur les sociétés",
                                               year, method="bm25")
                self.assertTrue(result["results"])
                self.assertEqual(result["edition_filter"], "on")
                self.assertTrue(all(row["edition"] == year for row in result["results"]))
        with self.assertRaisesRegex(ValueError, "obligatoire"):
            self.retriever.search("Article premier", 2025, edition_filter="off")

    def test_demonstration_artifacts(self):
        for path in ("demo.ipynb", "docs/03_evaluation.pdf",
                     "data/evaluation/campagne_principale/final_summary.json",
                     "data/evaluation/test_supplementaire/retrieval_summary.json"):
            self.assertTrue((ROOT / path).is_file(), path)
        notebook = json.loads((ROOT / "demo.ipynb").read_text(encoding="utf-8"))
        self.assertTrue(any(cell["cell_type"] == "code" for cell in notebook["cells"]))


if __name__ == "__main__":
    unittest.main()
