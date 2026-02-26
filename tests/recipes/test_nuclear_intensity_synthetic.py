from __future__ import annotations

import numpy as np

from ezquant.recipes.nuclear_intensity import NuclearIntensityRecipe


def test_nuclear_intensity_synthetic_recovery():
    image = np.full((64, 64), 10.0, dtype=np.float32)
    yy, xx = np.ogrid[:64, :64]
    circle = (yy - 32) ** 2 + (xx - 32) ** 2 < 10 ** 2
    image[circle] = 110.0

    masks = np.zeros((64, 64), dtype=np.int32)
    masks[circle] = 1

    recipe = NuclearIntensityRecipe()
    raw = recipe.run(inputs={"image": image}, params=recipe.get_default_params(), context={"masks": masks})
    row = raw["rows"][0]
    assert abs(row["mean_intensity_bg_corrected"] - 100.0) < 10.0
