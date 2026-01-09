"""
Wrapper for matplotlib.pyplot
"""

from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as _plt
import numpy as _np
from matplotlib.collections import PathCollection
from numpy.typing import ArrayLike


def scatter(
    x: ArrayLike,
    y: ArrayLike,
    *args: Any,
    point_labels: Iterable[str] | None = None,  # add labels/annotation to points
    **kwargs: Any,
) -> PathCollection:
    result = _plt.scatter(x, y, *args, **kwargs)
    if point_labels is not None:
        x_arr, y_arr = _np.asarray(x), _np.asarray(y)
        for xi, yi, point_label in zip(x_arr, y_arr, point_labels, strict=True):
            _plt.annotate(point_label, (xi, yi))
    return result


_CUSTOM_FUNCTIONS: dict[str, Any] = {
    "scatter": scatter,
}


def __getattr__(name: str) -> Any:
    if name in _CUSTOM_FUNCTIONS:
        return _CUSTOM_FUNCTIONS[name]
    return getattr(_plt, name)


def __dir__() -> list[str]:
    return list(_CUSTOM_FUNCTIONS.keys()) + dir(_plt)
