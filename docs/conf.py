"""Sphinx configuration. The options page is generated from the solver's own
option table (atrfu.solver.GROUPS and DOCS), so it always matches the code."""
from pathlib import Path

import atrfu
from atrfu.solver import CONFIG, DOCS, GROUPS

project = 'atrfu'
author = 'Gul Hameed'
copyright = '2026, Gul Hameed'
release = version = atrfu.__version__

extensions = ['myst_parser', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon',
              'sphinx.ext.mathjax', 'sphinx.ext.viewcode']
myst_enable_extensions = ['deflist', 'dollarmath']
myst_heading_anchors = 3
exclude_patterns = ['_build']
html_theme = 'furo'
html_title = f'atrfu {release}'
autodoc_member_order = 'bysource'


def write_options_page(app):
    lines = ['# Options reference', '',
             'Every option of `TrustRegionSolver`, with its default. Pass options as',
             'keyword arguments; names may use underscores or spaces. The same list is',
             'printed by `TrustRegionSolver.options()`. See the [tuning guide](tuning.md)',
             'for which options matter and how to set them.', '']
    for group, names in GROUPS.items():
        lines += [f'## {group}', '']
        for name in names:
            default = '{}' if name == 'solver_options' else repr(CONFIG.get(name).value())
            lines += [f"`{name.replace(' ', '_')}` = `{default}`",
                      f": {DOCS[name].replace('*', chr(92) + '*')}", '']
    page = Path(app.srcdir) / 'options.md'
    text = '\n'.join(lines)
    if not page.exists() or page.read_text(encoding='utf-8') != text:
        page.write_text(text, encoding='utf-8')


def setup(app):
    app.connect('builder-inited', write_options_page)
