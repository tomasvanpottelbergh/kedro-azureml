import logging
import re
from typing import Dict, Optional

from azure.ai.ml import Input, Output, command
from azure.ai.ml.dsl import pipeline as azure_pipeline
from azure.ai.ml.entities import Job
from azure.ai.ml.entities._builders import Command
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node

from kedro_azureml.config import KedroAzureMLConfig
from kedro_azureml.data import AzureMLDataSet

logger = logging.getLogger(__name__)


class AzureMLPipelineGenerator:
    def __init__(
        self,
        pipeline_name: str,
        kedro_environment: str,
        config: KedroAzureMLConfig,
        catalog: DataCatalog,
        docker_image: Optional[str] = None,
        params: Optional[str] = None,
        load_version: Dict[str, str] = {},
    ):
        self.kedro_environment = kedro_environment
        self.params = params
        self.catalog = catalog
        self.docker_image = docker_image
        self.config = config
        self.pipeline_name = pipeline_name
        self.load_versions = load_version

    def generate(self) -> Job:
        pipeline = self.get_kedro_pipeline()

        logger.info(f"Translating {self.pipeline_name} to Azure ML Pipeline")

        def kedro_azure_pipeline_fn():
            commands = {}

            for node in pipeline.nodes:
                azure_command = self._construct_azure_command(pipeline, node)

                commands[node.name] = azure_command

            # wire the commands into execution graph
            invoked_components = self._connect_commands(pipeline, commands)

            # pipeline outputs
            azure_pipeline_outputs = self._gather_pipeline_outputs(
                pipeline, invoked_components
            )
            return azure_pipeline_outputs

        kedro_azure_pipeline = azure_pipeline(name=self.pipeline_name)(
            kedro_azure_pipeline_fn
        )

        azure_pipeline_job: Job = kedro_azure_pipeline()
        return azure_pipeline_job

    def get_kedro_pipeline(self) -> Pipeline:
        from kedro.framework.project import pipelines

        pipeline: Pipeline = pipelines[self.pipeline_name]
        return pipeline

    def _sanitize_param_name(self, param_name: str) -> str:
        return re.sub(r"[^a-z0-9_]", "_", param_name.lower())

    def _sanitize_azure_name(self, name: str) -> str:
        return name.lower().replace(".", "__")

    def _construct_azure_command(
        self,
        pipeline: Pipeline,
        node: Node,
    ):
        # TODO - config can probably expose compute-per-step setting, to allow different steps to be scheduled on different machine types # noqa
        return command(
            name=self._sanitize_azure_name(node.name),
            display_name=node.name,
            command=self._prepare_command(node),
            environment=self.config.azure.environment_name,  # TODO: enable creation using Docker image
            inputs={
                self._sanitize_param_name(name): (
                    Input(type="string")
                    if (name in pipeline.inputs() and "params:" in name)
                    else Input()
                )
                for name in node.inputs
            },
            outputs={
                self._sanitize_param_name(name): Output() for name in node.outputs
            },
            code=".",
        )

    def _gather_pipeline_outputs(self, pipeline: Pipeline, invoked_components):
        azure_pipeline_outputs = {}
        for pipeline_output in pipeline.outputs():
            sanitized_output_name = self._sanitize_param_name(pipeline_output)
            source_node = next(
                (n for n in pipeline.nodes if pipeline_output in n.outputs), None
            )
            assert (
                source_node is not None
            ), f"There is no node which outputs `{pipeline_output}` dataset"
            azure_pipeline_outputs[sanitized_output_name] = invoked_components[
                source_node.name
            ].outputs[sanitized_output_name]
        return azure_pipeline_outputs

    def _get_versioned_name(self, dataset_name: str):
        version = self.load_versions.get(dataset_name)
        if version is None or version == "latest":
            versioned_name = f"{dataset_name}@latest"
        else:
            versioned_name = f"{dataset_name}:{version}"
        return versioned_name

    def _connect_commands(self, pipeline: Pipeline, commands: Dict[str, Command]):
        """
        So far, only standalone commands were constructed, this method
        connects command inputs with command outputs, to build the actual execution graph.
        Connection is done by "invoking" the commands, so the Azure's DSL builds them here
        :param pipeline:
        :param commands:
        :return:
        """
        node_deps = pipeline.node_dependencies
        invoked_components = {}
        for node in pipeline.nodes:  # pipeline.nodes are sorted topologically
            dependencies = node_deps[node]
            azure_inputs = {}
            for node_input in node.inputs:
                # 1. try to find output in dependencies
                sanitized_input_name = self._sanitize_param_name(node_input)
                output_from_deps = next(
                    (d for d in dependencies if node_input in d.outputs), None
                )
                if output_from_deps:
                    parent_outputs = invoked_components[output_from_deps.name].outputs
                    azure_output = parent_outputs[sanitized_input_name]
                    azure_inputs[sanitized_input_name] = azure_output
                elif node_input in self.catalog.list() and "params:" not in node_input:
                    # 2. try to find dataset in catalog
                    if isinstance(
                        ds := self.catalog._get_dataset(node_input), AzureMLDataSet
                    ):
                        azure_inputs[sanitized_input_name] = Input(
                            # TODO: report bug that @latest doesn't work with azureml: prefix
                            path=self._get_versioned_name(ds._name)
                        )
                    else:
                        raise TypeError(
                            f"Dataset {node_input} is not an AzureMLDataSet. "
                            "Only AzureMLDataSet is supported as an input to AzureMLPipeline."
                        )
                else:
                    # 3. if not found, provide dummy input
                    azure_inputs[sanitized_input_name] = node_input
            invoked_components[node.name] = commands[node.name](**azure_inputs)
        return invoked_components

    def _prepare_command(self, node):
        input_data_paths = (
            [
                f"--data-path={name} "
                + "${{inputs."
                + self._sanitize_param_name(name)
                + "}}"
                for name in node.inputs
                if "params:" not in name
            ]
            if node.inputs
            else []
        )
        output_data_paths = (
            [
                f"--data-path={name} "
                + "${{outputs."
                + self._sanitize_param_name(name)
                + "}}"
                for name in node.outputs
            ]
            if node.outputs
            else []
        )
        return (
            f"kedro azureml -e {self.kedro_environment} execute --pipeline={self.pipeline_name} --node={node.name} "  # noqa
            + " ".join(input_data_paths + output_data_paths)
            + (f" --params='{self.params}'" if self.params else "")
        ).strip()
