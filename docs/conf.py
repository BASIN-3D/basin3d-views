# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import subprocess

release = "0.noversion"
try:
    from basin3d_views.version import __release__  # type: ignore

    release = __release__
except ImportError:
    try:
        release = subprocess.check_output(["git", "describe", "--tags"]).rstrip().decode('utf-8')
    except Exception:
        pass
version = release.split("-")[0]

project = 'BASIN-3D Views Library'
copyright = '2022, Danielle S. Christianson, Valerie C. Hendrix, Catherin Wong'
author = 'Danielle S. Christianson, Valerie C. Hendrix, Catherin Wong'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
]
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'basin3d': ('https://basin3d.readthedocs.io/en/develop', None),
                       'django-basin3d': ('https://basin3d.readthedocs.io/projects/django-basin3d/en/latest/', None)}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
