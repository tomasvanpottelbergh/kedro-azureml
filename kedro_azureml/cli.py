import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import click
from kedro.framework.startup import ProjectMetadata

from kedro_azureml.cli_functions import (
    get_context_and_pipeline,
    parse_extra_params,
)
from kedro_azureml.client import AzureMLPipelinesClient
from kedro_azureml.config import CONFIG_TEMPLATE_YAML
from kedro_azureml.constants import (
    FILL_IN_ENVIRONMENT_NAME,
    KEDRO_AZURE_BLOB_TEMP_DIR_NAME,
)
from kedro_azureml.runner import AzurePipelinesRunner
from kedro_azureml.utils import CliContext, KedroContextManager

logger = logging.getLogger(__name__)


@click.group("AzureML")
def commands():
    """Kedro plugin adding support for Azure ML Pipelines"""
    pass


@commands.group(
    name="azureml", context_settings=dict(help_option_names=["-h", "--help"])
)
@click.option(
    "-k",
    "--kedro-env",
    type=str,
    default=lambda: os.environ.get("KEDRO_ENV", "local"),
    help="Kedro environment to use.",
)
@click.pass_obj
@click.pass_context
def azureml_group(ctx, metadata: ProjectMetadata, kedro_env):
    click.echo(metadata)
    ctx.obj = CliContext(kedro_env, metadata)


@azureml_group.command()
@click.argument("resource_group")
@click.argument("workspace_name")
@click.argument("subscription_id")
@click.argument("experiment_name")
@click.argument("cluster_name")
@click.argument("storage_account_name")
@click.argument("storage_container")
@click.pass_obj
def init(
    ctx: CliContext,
    resource_group,
    workspace_name,
    subscription_id,
    experiment_name,
    cluster_name,
    storage_account_name,
    storage_container,
):
    """
    Creates basic configuration for Kedro AzureML plugin
    """
    with KedroContextManager(ctx.metadata.package_name, ctx.env) as _:
        target_path = Path.cwd().joinpath("conf/base/azureml.yml")
        cfg = CONFIG_TEMPLATE_YAML.format(
            **{
                "resource_group": resource_group,
                "workspace_name": workspace_name,
                "subscription_id": subscription_id,
                "experiment_name": experiment_name,
                "cluster_name": cluster_name,
                "environment_name": FILL_IN_ENVIRONMENT_NAME,
                "storage_container": storage_container,
                "storage_account_name": storage_account_name,
            }
        )
        target_path.write_text(cfg)

        click.echo(f"Configuration generated in {target_path}")

        click.echo(
            click.style(
                f"It's recommended to set Lifecycle management rule for storage container {storage_container} "
                f"to avoid costs of long-term storage of the temporary data."
                f"\nTemporary data will be stored under abfs://{storage_container}/{KEDRO_AZURE_BLOB_TEMP_DIR_NAME} path"  # noqa
                f"\nSee https://docs.microsoft.com/en-us/azure/storage/blobs/lifecycle-management-policy-configure?tabs=azure-portal",  # noqa
                fg="green",
            )
        )


@azureml_group.command()
@click.option(
    "-e",
    "--environment",
    type=str,
    help="Azure ML Environment to use for pipeline execution.",
)
@click.option(
    "-p",
    "--pipeline",
    "pipeline",
    type=str,
    help="Name of pipeline to run",
    default="__default__",
)
@click.option(
    "--params",
    "params",
    type=str,
    help="Parameters override in form of JSON string",
)
@click.option("--wait-for-completion", type=bool, is_flag=True, default=False)
@click.pass_obj
@click.pass_context
def run(
    click_context: click.Context,
    ctx: CliContext,
    environment: Optional[str],
    pipeline: str,
    params: str,
    wait_for_completion: bool,
):
    """Runs the specified pipeline in Azure ML Pipelines; Additional parameters can be passed from command line.
    Can be used with --wait-for-completion param to block the caller until the pipeline finishes in Azure ML.
    """
    params = json.dumps(p) if (p := parse_extra_params(params)) else ""

    if environment:
        click.echo(f"Overriding Environment for run to: {environment}")

    mgr: KedroContextManager
    with get_context_and_pipeline(ctx, environment, pipeline, params) as (
        mgr,
        az_pipeline,
    ):
        az_client = AzureMLPipelinesClient(az_pipeline)

        is_ok = az_client.run(
            mgr.plugin_config.azure,
            wait_for_completion,
            lambda job: click.echo(job.studio_url),
        )

        if is_ok:
            exit_code = 0
            click.echo(
                click.style(
                    "Pipeline {} successfully".format(
                        "finished" if wait_for_completion else "started"
                    ),
                    fg="green",
                )
            )
        else:
            exit_code = 1
            click.echo(
                click.style("There was an error while running the pipeline", fg="red")
            )

        click_context.exit(exit_code)


@azureml_group.command()
@click.option(
    "-e",
    "--environment",
    type=str,
    help="Azure ML Environment to use for pipeline execution.",
)
@click.option(
    "-p",
    "--pipeline",
    "pipeline",
    type=str,
    help="Name of pipeline to run",
    default="__default__",
)
@click.option(
    "--params",
    "params",
    type=str,
    help="Parameters override in form of JSON string",
)
@click.option(
    "-o",
    "--output",
    type=click.types.Path(exists=False, dir_okay=False),
    default="pipeline.yaml",
    help="Pipeline YAML definition file.",
)
@click.pass_obj
def compile(
    ctx: CliContext,
    environment: Optional[str],
    pipeline: str,
    params: list,
    output: str,
):
    """Compiles the pipeline into YAML format"""
    params = json.dumps(p) if (p := parse_extra_params(params)) else ""
    with get_context_and_pipeline(ctx, environment, pipeline, params) as (
        _,
        az_pipeline,
    ):
        Path(output).write_text(str(az_pipeline))
        click.echo(f"Compiled pipeline to {output}")


@azureml_group.command(hidden=True)
@click.option(
    "-p",
    "--pipeline",
    "pipeline",
    type=str,
    help="Name of pipeline to run",
    default="__default__",
)
@click.option(
    "-n", "--node", "node", type=str, help="Name of the node to run", required=True
)
@click.option(
    "--params",
    "params",
    type=str,
    help="Parameters override in form of `key=value`",
)
@click.option(
    "--az-output",
    "azure_outputs",
    type=str,
    multiple=True,
    help="Paths of Azure ML Pipeline outputs to save dummy data into",
)
@click.pass_obj
def execute(
    ctx: CliContext, pipeline: str, node: str, params: str, azure_outputs: List[str]
):
    # 1. Run kedro
    parameters = parse_extra_params(params)
    with KedroContextManager(
        ctx.metadata.package_name, env=ctx.env, extra_params=parameters
    ) as mgr:
        runner = AzurePipelinesRunner()
        mgr.session.run(pipeline, node_names=[node], runner=runner)

    # 2. Save dummy outputs
    for dummy_output in azure_outputs:
        (Path(dummy_output) / "output.txt").write_text("#getindata")
