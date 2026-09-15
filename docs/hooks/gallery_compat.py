"""Restore ``ast.Str`` for mkdocs-gallery on Python 3.12+.

mkdocs-gallery 0.10.4 still does ``isinstance(node, ast.Str)`` and reads
``node.s``, both removed from the ``ast`` module in Python 3.12
(https://github.com/smarie/mkdocs-gallery/issues/108, open/unfixed). This
hook runs on mkdocs startup, before the gallery plugin is imported, and
patches just enough of the old API back in. Remove once upstream ships a
fix for Python 3.12+.
"""

import ast


def on_startup(**_kwargs: object) -> None:
    if not hasattr(ast, "Str"):
        ast.Str = ast.Constant
        ast.Constant.s = property(lambda self: self.value)
