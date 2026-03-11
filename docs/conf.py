from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

project = "WeightsLab"
author = "Graybox"
copyright = f"{datetime.now().year}, {author}"


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _docs_release() -> str:
    docs_version = os.environ.get("DOCS_VERSION", "").strip()
    if docs_version:
        return docs_version

    docs_ref = os.environ.get("DOCS_REF", "").strip()
    if docs_ref:
        return docs_ref[1:] if docs_ref.startswith("v") else docs_ref

    tag = _git(["describe", "--tags", "--exact-match"])
    if tag:
        return tag[1:] if tag.startswith("v") else tag

    branch = _git(["branch", "--show-current"])
    if branch == "main":
        return "latest"
    if branch:
        return branch

    return "latest"


release = _docs_release()
version = release

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torchaudio",
    "torchmetrics",
    "pytorch_lightning",
    "grpc",
    "grpcio",
    "onnx",
    "onnxruntime",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"WeightsLab {release}"
html_logo = "_static/logo-light.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["version-switcher.js"]
html_theme_options = {
    "sidebar_hide_name": False,
}

# Build only the desired refs when using sphinx-multiversion.
smv_branch_whitelist = r"^(main|dev)$"
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_remote_whitelist = r"^origin$"

myst_heading_anchors = 3
