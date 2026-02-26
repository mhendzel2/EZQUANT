from __future__ import annotations

from ezquant.core.registry import get_recipe_registry


REQUIRED_METHODS = [
    "describe",
    "get_default_params",
    "preflight_requirements",
    "run",
    "postflight_checks",
    "build_outputs",
]


def test_recipe_contract_methods_present():
    registry = get_recipe_registry()
    for _, cls in registry.items():
        recipe = cls()
        for name in REQUIRED_METHODS:
            assert hasattr(recipe, name)


def test_recipe_declares_required_metadata():
    registry = get_recipe_registry()
    for _, cls in registry.items():
        req = cls().preflight_requirements()
        assert "required_metadata" in req
