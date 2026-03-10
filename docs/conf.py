from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

project = "WeightsLab"
author = "Graybox"
copyright = f"{datetime.now().year}, {author}"
docs_ref = os.environ.get("DOCS_REF", "")
release = os.environ.get("DOCS_VERSION", docs_ref if docs_ref else "0.0.0")
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
html_theme_options = {
    "sidebar_hide_name": False,
}

myst_heading_anchors = 3
