"""Separate simultaneously observed surface relief from temporal height drift."""
import numpy as np
from .legacy.semantic_grid import LayeredSemanticGridMap


class SurfaceGrid(LayeredSemanticGridMap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surface_height_range = np.zeros(self.elevation_count.shape, dtype=np.float32)

    def _observe_elevation_batch(self, row, column, minimum, maximum):
        # One update is one scan. A common height offset cannot add relief.
        # Keep previously observed relief when a later beam sees only one face.
        # Only confirmed free-space cleanup may reduce this historical evidence.
        self.surface_height_range[row, column] = max(
            self.surface_height_range[row, column], maximum - minimum)

    def height_range_layer(self):
        return np.where(self.elevation_count > 0, self.surface_height_range,
                        np.nan).astype(np.float32)
