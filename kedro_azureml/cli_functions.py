import json
from contextlib import contextmanager

import click

from kedro_azureml.generator import AzureMLPipelineGenerator
from kedro_azureml.utils import CliContext, KedroContextManager


@contextmanager
def get_context_and_pipeline(ctx: CliContext, image: str, pipeline: str, params):
    with KedroContextManager(ctx.metadata.package_name, ctx.env) as mgr:

        generator = AzureMLPipelineGenerator(
            pipeline,
            ctx.env,
            mgr.plugin_config,
            mgr.context.catalog,
            image,
            params,
        )
        az_pipeline = generator.generate()
        yield mgr, az_pipeline


def parse_extra_params(params):
    if params and (parameters := json.loads(params.strip("'"))):
        click.echo(
            f"Running with extra parameters:\n{json.dumps(parameters, indent=4)}"
        )
    else:
        parameters = None
    return parameters
