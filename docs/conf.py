import sys
import os

sys.path.insert(0, os.path.abspath(".."))

project = "CardamomOT"
copyright = "2026, Elias Ventre, Yann Mauge"
author = "Elias Ventre, Yann Mauge"
release = "2.0.0"

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "nbsphinx",
]

# -- MyST configuration -------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

# -- sphinx-autoapi -----------------------------------------------------------
autoapi_dirs = ["../CardamomOT"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_ignore = ["*cli*.py", "*logging*.py"]
autoapi_add_toctree_entry = False
autoapi_keep_files = True

# -- nbsphinx -----------------------------------------------------------------
nbsphinx_execute = "never"  # notebooks are pre-executed and committed with outputs
nbsphinx_allow_errors = False

# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
}

# -- General ------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "CardamomOT"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
