from unittest.mock import Mock

import pytest
from kedro.io.core import Version

from kedro_azureml.hooks import azureml_local_run_hook


@pytest.mark.parametrize(
    "runner",
    [("kedro_azureml.runner.AzurePipelinesRunner"), ("kedro.runner.SequentialRunner")],
)
def test_hook_after_context_created(
    mock_azureml_config, dummy_pipeline, multi_catalog, runner
):
    context_mock = Mock()
    context_mock.config_loader.get.return_value = {
        "azure": mock_azureml_config.to_dict()
    }

    azureml_local_run_hook.after_context_created(context_mock)
    assert azureml_local_run_hook.azure_config.subscription_id == "123"
    assert azureml_local_run_hook.azure_config.workspace_name == "best"

    run_params = {"runner": runner}

    azureml_local_run_hook.before_pipeline_run(
        run_params, dummy_pipeline, multi_catalog
    )

    if runner == "kedro.runner.SequentialRunner":
        assert multi_catalog.datasets.input_data._download == True
        assert multi_catalog.datasets.input_data._local_run == True
        assert (
            multi_catalog.datasets.input_data._azureml_config
            == azureml_local_run_hook.azure_config
        )
        assert multi_catalog.datasets.i2._download == False
        assert multi_catalog.datasets.i2._local_run == True
        assert multi_catalog.datasets.i2._version == Version("local", "local")
    else:
        assert multi_catalog.datasets.input_data._download == False
        assert multi_catalog.datasets.input_data._local_run == False
        assert multi_catalog.datasets.input_data._azureml_config is None
        assert multi_catalog.datasets.i2._download == False
        assert multi_catalog.datasets.i2._local_run == False
        assert multi_catalog.datasets.i2._version == Version(None, None)