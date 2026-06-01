from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import RepositoryConfig
from .models import SourceDocument


TEXT_EXTENSIONS = {
    ".cfg",
    ".css",
    ".go",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class IndexedFile:
    path: str
    size: int
    extension: str
    text: str


@dataclass
class RepositoryIndex:
    root: Path
    files: list[IndexedFile] = field(default_factory=list)

    @property
    def languages(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for item in self.files:
            counts[item.extension or "[no extension]"] += 1
        return counts


class RepositoryAnalyzer:
    def __init__(self, config: RepositoryConfig) -> None:
        self.config = config

    def index(self, root: str | os.PathLike[str]) -> RepositoryIndex:
        root_path = Path(root).resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {root_path}")

        files: list[IndexedFile] = []
        for path in root_path.rglob("*"):
            if not path.is_file() or self._is_ignored(path, root_path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > self.config.max_file_bytes:
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            files.append(
                IndexedFile(
                    path=str(path.relative_to(root_path)),
                    size=stat.st_size,
                    extension=path.suffix.lower(),
                    text=text,
                )
            )
        return RepositoryIndex(root=root_path, files=sorted(files, key=lambda item: item.path))

    def search(self, index: RepositoryIndex, query: str, *, limit: int = 10) -> list[IndexedFile]:
        tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9_./-]{3,}", query)]
        scored: list[tuple[int, IndexedFile]] = []
        for item in index.files:
            haystack = f"{item.path}\n{item.text[:12000]}".lower()
            score = sum(haystack.count(token) for token in tokens)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].path))
        return [item for _score, item in scored[:limit]]

    def dependencies(self, index: RepositoryIndex) -> dict[str, list[str]]:
        dependencies: dict[str, set[str]] = defaultdict(set)
        for item in index.files:
            if item.extension == ".py":
                for match in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z0-9_.]+)", item.text, re.MULTILINE):
                    dependencies[item.path].add(match.group(1).split(".")[0])
            elif item.extension in {".js", ".jsx", ".ts", ".tsx"}:
                for match in re.finditer(r"from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\)", item.text):
                    dependencies[item.path].add((match.group(1) or match.group(2)).split("/")[0])
            elif item.path.endswith("pyproject.toml"):
                for match in re.finditer(r"^\s*([A-Za-z0-9_.-]+)\s*[<>=~!]", item.text, re.MULTILINE):
                    dependencies[item.path].add(match.group(1))
        return {path: sorted(values) for path, values in sorted(dependencies.items())}

    def risks(self, index: RepositoryIndex) -> list[tuple[str, str]]:
        patterns = {
            "todo": r"\b(?:TODO|FIXME|HACK)\b",
            "dynamic_code": r"\b(?:eval|exec)\s*\(",
            "shell_execution": r"shell\s*=\s*True|subprocess\.",
            "unsafe_pickle": r"pickle\.load|pickle\.loads",
            "possible_secret": r"(?i)(api[_-]?key|secret|token|password)\s*[:=]",
        }
        findings: list[tuple[str, str]] = []
        for item in index.files:
            for label, pattern in patterns.items():
                if re.search(pattern, item.text):
                    findings.append((item.path, label))
        return findings

    def architecture_summary(self, index: RepositoryIndex) -> str:
        top_dirs: Counter[str] = Counter()
        for item in index.files:
            first = item.path.split(os.sep)[0] if os.sep in item.path else item.path.split("/")[0]
            top_dirs[first] += 1
        language_summary = ", ".join(
            f"{extension}: {count}" for extension, count in index.languages.most_common(8)
        )
        directory_summary = ", ".join(
            f"{name}: {count}" for name, count in top_dirs.most_common(8)
        )
        return (
            f"Indexed {len(index.files)} text files. "
            f"Languages/extensions: {language_summary or 'none'}. "
            f"Top paths: {directory_summary or 'none'}."
        )

    def sources_for_files(self, files: list[IndexedFile]) -> list[SourceDocument]:
        return [
            SourceDocument(
                id=item.path,
                title=item.path,
                url=item.path,
                text=item.text,
                source_type="repository",
                path=item.path,
            )
            for item in files
        ]

    def report(self, index: RepositoryIndex, question: str) -> tuple[str, list[SourceDocument]]:
        relevant = self.search(index, question, limit=12)
        if not relevant:
            relevant = index.files[:12]
        dependencies = self.dependencies(index)
        risks = self.risks(index)
        files_by_path = {item.path: item for item in index.files}
        cited_paths = {item.path for item in relevant[:12]}

        lines = [
            "# Repository Research Report",
            "",
            f"**Question:** {question}",
            "",
            "## Architecture Summary",
            self.architecture_summary(index),
            "",
            "## Relevant Files",
            "",
        ]
        for item in relevant[:12]:
            lines.append(f"- `{item.path}` ({item.size} bytes) [{item.path}]")
        lines.append("")

        if dependencies:
            lines.extend(["## Dependency Signals", ""])
            for path, deps in list(dependencies.items())[:12]:
                cited_paths.add(path)
                lines.append(f"- `{path}` imports or declares: {', '.join(deps[:12])} [{path}]")
            lines.append("")

        if risks:
            lines.extend(["## Risk Signals", ""])
            for path, label in risks[:20]:
                cited_paths.add(path)
                lines.append(f"- `{label}` pattern found in `{path}` [{path}]")
            lines.append("")
        else:
            lines.extend(["## Risk Signals", "No common static risk patterns were detected.", ""])

        lines.extend(
            [
                "## Read-only Scope",
                "Repository mode only indexes and summarizes files; it does not modify the repository.",
                "",
            ]
        )
        sources = self.sources_for_files(
            [files_by_path[path] for path in sorted(cited_paths) if path in files_by_path]
        )
        return "\n".join(lines).strip() + "\n", sources

    def _is_ignored(self, path: Path, root: Path) -> bool:
        try:
            relative = path.relative_to(root)
        except ValueError:
            return True
        return any(part in self.config.ignore_dirs for part in relative.parts)
