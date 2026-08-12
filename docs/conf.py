# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_provider_options import generate

generate()

project = "TensorRT RTX Execution Provider"
copyright = f"{datetime.now().year}, NVIDIA CORPORATION & AFFILIATES"  # noqa: A001

extensions = ["sphinx_rtd_theme", "myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_heading_anchors = 3
master_doc = "index"
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = "TensorRT RTX Execution Provider"
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = ["nvidia.css"]

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 2,
    "includehidden": True,
    "titles_only": False,
    "style_external_links": True,
    "style_nav_header_background": "#76b900",
}
