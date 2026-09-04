#!/usr/bin/env python3
"""Regenerate ``docs/whats_new.rst`` from the repository's GitHub releases.

The page is committed as static RST rather than fetched while Sphinx runs: a
docs build that reached out to api.github.com would fail on an offline machine,
in an air-gapped CI runner, or simply when the unauthenticated rate limit is
spent -- and it would fail at the moment someone is trying to build the docs,
which is the worst time to discover it. Regenerating is a deliberate step::

    python docs/_scripts/update_whats_new.py

Only **stable** releases are listed. Every tag push also publishes a
``vX.Y.Z.devN`` pre-release to TestPyPI for verification (see ci.yml's
``build-and-publish-dev-release``), and those outnumber the real releases
roughly two to one -- a "What's New" page carrying them would be mostly noise
about build plumbing.

Each summary is the release body **between the "Welcome" heading and the
"What's Changed precisely:" heading**, which is where the release template puts
the human description. Two transformations make that readable:

* the boilerplate wrapping it (identical in all ten releases) is stripped;
* the template flattens every PR description onto a single line, so the
  structure it had as markdown survives only as inline ``" - "`` separators and
  stray ``"## Heading"`` markers -- ``_description_lines`` recovers it as real
  nested bullets. Without that step the page is ten paragraphs of run-on text.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "GrayboxTech/weightslab"
RELEASE_COUNT = 10
OUTPUT = Path(__file__).resolve().parents[1] / "whats_new.rst"

LEAD_IN = re.compile(
    r"^Welcome to the \*\*[^*]+\*\* release of WeightsLab\.[^\n]*\n+", re.M)
TAIL = re.compile(r"\n*One platform\. Unmatched flexibility\..*\Z", re.S)

SECTION_MARKER = re.compile(r"\s*#{2,4}\s+([A-Z][^#\n]{0,40}?)\s+(?=[-*`\[A-Z])")
TOPLEVEL_HEADING = re.compile(r"#{2,6}\s+([^\n#]+?)\s*$", re.M)
ITEM_SPLIT = re.compile(r"\s+-\s+(?=[\[`*\w])")
PR_BULLET = re.compile(r"^\[([^\]]+)\]\(([^)]+)\)\s*:?\s*(.*)$", re.S)


def fetch_releases() -> list:
    releases, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{REPO}/releases?per_page=100&page={page}"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            batch = json.load(resp)
        releases += batch
        if len(batch) < 100:
            return releases
        page += 1


def stable_releases(releases: list) -> list:
    out = [r for r in releases if not r.get("prerelease") and not r.get("draft")]
    out.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    return out[:RELEASE_COUNT]


def welcome_section(body: str) -> str:
    match = re.search(
        r"^#{1,4}\s*Welcome\s*$(.*?)^#{1,4}\s*What's Changed", body or "", re.M | re.S)
    return (match.group(1) if match else "").strip()


def _strip_boilerplate(text: str) -> str:
    return TAIL.sub("", LEAD_IN.sub("", text)).strip()


def md_inline_to_rst(text: str) -> str:
    """Convert the inline markdown that turns up in PR descriptions.

    Deliberately small: these are PR descriptions pasted into a release
    template, so they carry links, bold, and inline code and little else.
    """
    text = re.sub(r"`([^`]+)`", r"``\1``", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"`\1 <\2>`__", text)
    text = re.sub(r"_\(([^)]+)\)_", r"*(\1)*", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _sanitize(text: str) -> str:
    """Repair the two things the release template's flattening leaves behind.

    A GitHub task-list marker (``[ ]``) carries no meaning once the checklist
    has been flattened into prose, and an unbalanced backtick is an outright
    RST error -- which happens whenever the template truncates a long PR body
    mid-token, as it does with an ellipsis. Closing the text at the last
    balanced point is honest about the truncation and keeps the build clean;
    the card links to the full release notes anyway.
    """
    text = re.sub(r"^[-*]\s+", "", text.strip())
    text = re.sub(r"^\[[ xX]\]\s*", "", text)
    for marker in ("`", "**"):
        if text.count(marker) % 2:
            text = text[:text.rindex(marker)].rstrip(" .") + " …"
    return text.strip()


def _wrap(text: str, width: int, first: str, rest: str) -> list:
    """Greedy wrap that keeps an RST bullet or indent prefix on the first line."""
    words = text.split()
    if not words:
        return []
    out, line = [], first + words[0]
    for word in words[1:]:
        if len(line) + 1 + len(word) > width:
            out.append(line)
            line = rest + word
        else:
            line += " " + word
    out.append(line)
    return out


def _description_lines(description: str, indent: str) -> list:
    """Render one PR's flattened description as nested RST beneath its bullet."""
    if not description.strip():
        return []

    marked = SECTION_MARKER.sub(
        lambda m: "\n@@" + m.group(1).rstrip(":") + "@@\n", description)

    lines = []
    for block in marked.split("\n"):
        block = block.strip()
        if not block:
            continue
        heading = re.fullmatch(r"@@(.+)@@", block)
        if heading:
            lines.append("")
            lines.append(indent + "**" + md_inline_to_rst(heading.group(1)) + "**")
            continue
        items = [i.strip() for i in ITEM_SPLIT.split(block) if i.strip()]
        lines.append("")
        if len(items) > 1:
            for item in items:
                lines.extend(_wrap(_sanitize(md_inline_to_rst(item)), 86,
                                   first=indent + "- ", rest=indent + "  "))
        else:
            lines.extend(_wrap(_sanitize(md_inline_to_rst(items[0])), 86,
                               first=indent, rest=indent))
    return lines


def summary_to_rst(body: str) -> list:
    text = _strip_boilerplate(welcome_section(body))
    if not text:
        return ["   *No description was published with this release.*"]

    # Some releases write real ATX headers (``#### Foo``) straight into the
    # welcome section instead of the usual flattened "PR list" shape -- and
    # the flattening GitHub applies to the release template can glue one onto
    # the tail of the preceding sentence on the same line. In both cases the
    # heading is the remainder of its raw line, so marking off everything
    # from the hashes to end-of-line (multiline mode) catches both without
    # needing them on their own line first.
    marked = TOPLEVEL_HEADING.sub(
        lambda m: "\n\n@@" + m.group(1).strip().strip("*").strip() + "@@\n\n", text)

    lines, entries, current = [], [], None

    def flush_entries():
        for entry in entries:
            joined = " ".join(entry).strip()
            match = PR_BULLET.match(joined)
            if match:
                label, url, description = match.groups()
                lines.append("   - `" + label.strip() + " <" + url.strip() + ">`__")
                lines.extend(_description_lines(description, "     "))
            else:
                lines.extend(_wrap(_sanitize(md_inline_to_rst(joined)), 86, first="   - ", rest="     "))
            lines.append("")
        entries.clear()

    for raw in marked.splitlines():
        stripped = raw.strip()
        if not stripped or stripped == "---":
            continue
        heading = re.fullmatch(r"@@(.+)@@", stripped)
        if heading:
            # A heading always closes whatever bullet/list came before it, so
            # its entries are flushed (keeping them grouped under the right
            # heading) before the heading itself is emitted as bold text with
            # blank lines on both sides -- RST requires that gap around a
            # paragraph sitting next to a list.
            flush_entries()
            current = None
            lines.append("")
            lines.append("   **" + md_inline_to_rst(heading.group(1)) + "**")
            lines.append("")
            continue
        if stripped.startswith("- "):
            current = [stripped[2:]]
            entries.append(current)
        elif stripped.startswith(">") and current is not None:
            current.append(stripped.lstrip("> ").strip())
        elif current is not None:
            current.append(stripped)
        else:
            lines.extend(_wrap(_sanitize(md_inline_to_rst(stripped)), 86, first="   ", rest="   "))

    flush_entries()

    while lines and not lines[-1].strip():
        lines.pop()
    return lines


HEADER = """.. _whats-new:

What's New
==========

The last {count} stable releases of WeightsLab, newest first. Each card links to
its full release notes on GitHub, where the complete "What's Changed" commit
list lives.

.. note::

   Only stable releases appear here. Every tag also publishes a
   ``vX.Y.Z.devN`` pre-release to TestPyPI for verification; those are build
   plumbing rather than user-facing changes.

.. tip::

   Upgrade with ``pip install --upgrade weightslab``. Setting up for the first
   time? Start at :doc:`quickstart`.

"""

FOOTER = """
----

`Every release on GitHub → <https://github.com/{repo}/releases>`__

.. This page is generated. To refresh it after a release:
..     python docs/_scripts/update_whats_new.py
.. Last generated: {generated}
"""


def build_rst(releases: list) -> str:
    parts = [HEADER.format(count=len(releases))]
    for release in releases:
        published = datetime.fromisoformat(
            release["published_at"].replace("Z", "+00:00")).astimezone(timezone.utc)
        date_label = f"{published.strftime('%B')} {published.day}, {published.year}"

        parts.append(".. card::\n")
        parts.append(f"   :link: {release['html_url']}\n")
        parts.append("   :class-card: wl-release-card\n\n")
        parts.append("   .. rst-class:: wl-release-date\n\n")
        parts.append(f"   {date_label}\n\n")
        parts.append("   .. rst-class:: wl-release-version\n\n")
        parts.append(f"   {release['tag_name']}\n\n")
        parts.append("\n".join(summary_to_rst(release.get("body") or "")))
        parts.append("\n\n")
    parts.append(FOOTER.format(
        repo=REPO, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d")))
    return "".join(parts)


def main() -> None:
    releases = stable_releases(fetch_releases())
    OUTPUT.write_text(build_rst(releases), encoding="utf-8")
    print(f"wrote {OUTPUT} — {len(releases)} releases, "
          f"{releases[-1]['tag_name']} … {releases[0]['tag_name']}")


if __name__ == "__main__":
    main()
