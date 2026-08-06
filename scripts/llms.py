"""Build the complete LLM-readable documentation bundle."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import yaml

YamlValue: TypeAlias = None | bool | int | float | str | list["YamlValue"] | dict[str, "YamlValue"]

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "mkdocs.yml"
DOCS_PATH = ROOT / "docs"
OUTPUT_PATH = DOCS_PATH / "llms-full.txt"
STAGING_PATH = DOCS_PATH / "_markdown"
SITE_URL = "https://vcoderun.github.io/autobench"


@dataclass(frozen=True, slots=True)
class Page:
    """A public documentation page in navigation order."""

    title: str
    source: Path


def iter_pages(entries: list[YamlValue]) -> Iterator[Page]:
    """Yield leaf pages from a nested Zensical navigation tree."""
    for entry in entries:
        if isinstance(entry, str):
            source = Path(entry)
            yield Page(source.stem.replace("-", " ").title(), source)
            continue
        if not isinstance(entry, dict):
            raise ValueError("Every nav entry must be a page path or a named section.")
        for title, value in entry.items():
            if isinstance(value, str):
                yield Page(title, Path(value))
            elif isinstance(value, list):
                yield from iter_pages(value)
            else:
                raise ValueError(f"Navigation section {title!r} must contain pages or sections.")


def load_pages() -> list[Page]:
    """Load and validate all public documentation pages."""
    config = cast(YamlValue, yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    if not isinstance(config, dict):
        raise ValueError("mkdocs.yml must contain a mapping.")
    nav = config.get("nav")
    if not isinstance(nav, list):
        raise ValueError("mkdocs.yml must define nav as a list.")

    pages = list(iter_pages(nav))
    sources = [page.source for page in pages]
    if len(sources) != len(set(sources)):
        raise ValueError("Each documentation page may appear in nav only once.")

    missing = [str(page.source) for page in pages if not (DOCS_PATH / page.source).is_file()]
    if missing:
        raise ValueError(f"Navigation references missing pages: {', '.join(missing)}")

    public_sources = {
        path.relative_to(DOCS_PATH)
        for path in DOCS_PATH.rglob("*.md")
        if "plans" not in path.relative_to(DOCS_PATH).parts and not path.name.endswith("-plan.md")
    }
    unlisted = sorted(str(path) for path in public_sources.difference(sources))
    if unlisted:
        raise ValueError(f"Public documentation pages missing from nav: {', '.join(unlisted)}")
    return pages


def render_full_documentation(pages: list[Page]) -> str:
    """Render all public Markdown pages as one deterministic document."""
    sections = [
        "# Autobench Full Documentation",
        "",
        "> Complete, LLM-readable documentation for Autobench: define how an application should "
        "be measured, then run, record, compare, and replay its benchmarks in one consistent "
        "format.",
        "",
        f"Canonical documentation: {SITE_URL}/",
        "",
        "This file follows the site navigation order. Each section includes the canonical "
        "page URL followed by its complete Markdown source.",
    ]
    for page in pages:
        source = page.source.as_posix()
        route = "" if source == "index.md" else f"{source.removesuffix('.md')}/"
        markdown = (DOCS_PATH / page.source).read_text(encoding="utf-8").strip()
        sections.extend(
            [
                "",
                "---",
                "",
                f"## {page.title}",
                "",
                f"Canonical page: {SITE_URL}/{route}",
                "",
                markdown,
            ]
        )
    return "\n".join(sections) + "\n"


def stage_markdown_sources(pages: list[Page]) -> None:
    """Stage exact page sources as static documentation assets."""
    shutil.rmtree(STAGING_PATH, ignore_errors=True)
    for page in pages:
        destination = (STAGING_PATH / page.source).with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DOCS_PATH / page.source, destination)


def main() -> None:
    """Generate or verify the complete documentation bundle."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--stage", action="store_true")
    args = parser.parse_args()

    pages = load_pages()
    content = render_full_documentation(pages)
    if args.write:
        OUTPUT_PATH.write_text(content, encoding="utf-8")
        return
    if args.stage:
        stage_markdown_sources(pages)
        return
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != content:
        raise SystemExit("docs/llms-full.txt is stale; run `make docs-llms`.")


if __name__ == "__main__":
    main()
