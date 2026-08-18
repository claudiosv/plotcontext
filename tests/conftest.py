import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _close_figures():
    """Ensure every test starts and ends with a clean pyplot figure stack."""
    plt.close("all")
    yield
    plt.close("all")
