"""Minimal UI entry that delegates execution to core pipeline."""

from ezquant.core.pipeline import Pipeline


def run_app(policy_path: str, input_path: str, output_dir: str):
    pipe = Pipeline(policy_path)
    return pipe.run(
        input_path=input_path,
        recipe_name="nuclear_intensity",
        role="student",
        output_dir=output_dir,
    )
