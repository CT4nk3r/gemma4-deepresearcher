import unittest

from gemma_research.agent import ResearchAgent
from gemma_research.config import Config
from gemma_research.models import SearchResult, SourceDocument


class FakeSearcher:
    def search(self, query):
        return [
            SearchResult(query=query, title="Gemma overview", url="https://example.test/a", snippet="", rank=1),
            SearchResult(query=query, title="Local research", url="https://example.test/b", snippet="", rank=2),
        ]


class FakeReader:
    def read(self, result):
        return SourceDocument(
            id=f"S{result.rank}",
            title=result.title,
            url=result.url,
            text=(
                "Gemma local research agents benefit from deterministic orchestration. "
                "Local execution keeps state outside the model and cites collected sources."
            ),
        )


class AgentOfflineTests(unittest.TestCase):
    def test_offline_agent_produces_cited_report(self):
        config = Config.default()
        config.model.provider = "offline"
        config.agent.min_sources = 1
        agent = ResearchAgent(config, searcher=FakeSearcher(), reader=FakeReader())
        report = agent.research("How should Gemma local research work?", trace=False)
        self.assertIn("[S1]", report.markdown)
        self.assertTrue(report.sources)


if __name__ == "__main__":
    unittest.main()
