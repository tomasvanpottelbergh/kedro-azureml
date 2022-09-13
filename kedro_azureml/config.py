from typing import Optional

import yaml
from pydantic import BaseModel


class DockerConfig(BaseModel):
    image: str


class AzureMLConfig(BaseModel):
    experiment_name: str
    workspace_name: str
    resource_group: str
    cluster_name: str
    environment_name: str


class KedroAzureMLConfig(BaseModel):
    azure: AzureMLConfig
    docker: Optional[DockerConfig]


CONFIG_TEMPLATE_YAML = """
azure:
  # Name of the Azure ML Compute Cluster
  cluster_name: "{cluster_name}"
  # Azure ML Experiment Name
  experiment_name: "{experiment_name}"
  # Azure resource group to use
  resource_group: "{resource_group}"
  # Azure ML Workspace name
  workspace_name: "{workspace_name}"
  # Azure ML Environment to use during pipeline execution
  environment_name: "{environment_name}"

docker:
  # Docker image to use during pipeline execution
  image: "{docker_image}"
""".strip()

# This auto-validates the template above during import
_CONFIG_TEMPLATE = KedroAzureMLConfig.parse_obj(yaml.safe_load(CONFIG_TEMPLATE_YAML))
