import os
import sys
import tomllib
from datetime import date
from pathlib import Path

current_date = date.today()
current_year = current_date.year

sys.path.insert(0, os.path.abspath('..'))

with open("/code/pyproject.toml", "rb") as f:  # binary required to parse file as UTF-8 with universal newlines disabled
    project_metadata = tomllib.load(f)
project_name = project_metadata["tool"]["poetry"]["name"]
project_version = project_metadata["tool"]["poetry"]["version"]
project_author = project_metadata["tool"]["poetry"]["authors"][0]

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = project_name.title()
copyright = str(current_year)
author = project_author
release = project_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' # 'alabaster'
html_static_path = ['_static']
