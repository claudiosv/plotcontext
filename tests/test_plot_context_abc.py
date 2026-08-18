from pathlib import Path

import pytest

from plotcontext.plot_context import AbstractPlotContext


def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError):
        AbstractPlotContext()


def test_concrete_subclass_must_implement_all_abstract_methods():
    class Incomplete(AbstractPlotContext):
        @staticmethod
        def slides_rc_params():
            return {}

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_subclass_can_be_instantiated_and_used():
    class Minimal(AbstractPlotContext):
        def __init__(self):
            self.entered = False
            self.exited = False
            self.drawn = False
            self.closed = False

        @staticmethod
        def slides_rc_params():
            return {"figure.dpi": 300}

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._draw()
            self._close()
            self.exited = True
            return False

        def save(self, filename, **save_kwargs):
            self.saved = (filename, save_kwargs)

        def save_fig(self, output_path, name):
            self.saved_fig = (Path(output_path), name)

        def _draw(self):
            self.drawn = True

        def _close(self):
            self.closed = True

    with Minimal() as ctx:
        assert ctx.entered is True

    assert ctx.exited is True
    assert ctx.drawn is True
    assert ctx.closed is True
    assert Minimal.slides_rc_params() == {"figure.dpi": 300}

    ctx.save("out.pdf", dpi=300)
    assert ctx.saved == ("out.pdf", {"dpi": 300})

    ctx.save_fig("out_dir", "name")
    assert ctx.saved_fig == (Path("out_dir"), "name")
