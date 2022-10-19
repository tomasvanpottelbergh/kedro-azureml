from pathlib import Path
from typing import Any, Dict

from kedro.extras.datasets.pickle.pickle_dataset import PickleDataSet
from kedro.io import AbstractDataSet, DataCatalog
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from pluggy import PluginManager

from kedro_azureml.data import AzureMLInputDataSet


class AzurePipelinesRunner(SequentialRunner):
    def __init__(self, is_async: bool = False, data_paths: Dict[str, str] = dict()):
        super().__init__(is_async)
        self.data_paths = data_paths

    def run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager = None,
        session_id: str = None,
    ) -> Dict[str, Any]:
        catalog = catalog.shallow_copy()

        # Loop over input and output datasets in arguments to set their paths
        for ds_name, ds_path in self.data_paths.items():
            if ds_name in catalog.list():
                ds = catalog._get_dataset(ds_name)
                if isinstance(ds, AzureMLInputDataSet):
                    ds.convert_to_supertype()
                ds._filepath = Path(ds_path) / Path(ds._filepath).name
                ds._version = None
                catalog.add(ds_name, ds, replace=True)
            else:
                catalog.add(ds_name, self.create_default_data_set(ds_name))

        return super().run(pipeline, catalog, hook_manager, session_id)

    def create_default_data_set(self, ds_name: str) -> AbstractDataSet:
        return PickleDataSet(str(Path(self.data_paths[ds_name]) / f"{ds_name}.pickle"))
