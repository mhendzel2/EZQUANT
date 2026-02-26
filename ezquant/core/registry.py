from __future__ import annotations

from ezquant.core.recipe_base import RecipeBase
from ezquant.recipes.colocalization import ColocalizationRecipe
from ezquant.recipes.frap_basic import FrapBasicRecipe
from ezquant.recipes.nb_camera_calibrated import NBCameraCalibratedRecipe
from ezquant.recipes.nuclear_intensity import NuclearIntensityRecipe
from ezquant.recipes.puncta_counting import PunctaCountingRecipe


def get_recipe_registry() -> dict[str, type[RecipeBase]]:
    return {
        "nuclear_intensity": NuclearIntensityRecipe,
        "puncta_counting": PunctaCountingRecipe,
        "colocalization": ColocalizationRecipe,
        "frap_basic": FrapBasicRecipe,
        "nb_camera_calibrated": NBCameraCalibratedRecipe,
    }
