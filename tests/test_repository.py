import tempfile
import unittest
from pathlib import Path

from gemma_research.config import RepositoryConfig
from gemma_research.repository import RepositoryAnalyzer
from gemma_research.verification import validate_report_citations


class RepositoryAnalyzerTests(unittest.TestCase):
    def test_indexes_searches_dependencies_and_risks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "import json\n# TODO: review\nprint(json.dumps({'ok': True}))\n",
                encoding="utf-8",
            )
            analyzer = RepositoryAnalyzer(RepositoryConfig())
            index = analyzer.index(root)
            matches = analyzer.search(index, "json review")
            deps = analyzer.dependencies(index)
            risks = analyzer.risks(index)

        self.assertEqual(len(index.files), 1)
        self.assertEqual(matches[0].path, "app.py")
        self.assertIn("json", deps["app.py"])
        self.assertIn(("app.py", "todo"), risks)

    def test_report_includes_all_cited_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("import helper\nprint('main')\n", encoding="utf-8")
            (root / "helper.py").write_text("# TODO: improve\nVALUE = 1\n", encoding="utf-8")
            analyzer = RepositoryAnalyzer(RepositoryConfig())
            index = analyzer.index(root)
            markdown, sources = analyzer.report(index, "main architecture")

        self.assertTrue(validate_report_citations(markdown, sources).ok)


if __name__ == "__main__":
    unittest.main()
