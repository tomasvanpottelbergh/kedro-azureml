from pathlib import Path
from typing import Any, Dict

from kedro.extras.datasets.pickle.pickle_dataset import PickleDataSet
from kedro.io import AbstractDataSet, DataCatalog
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from pluggy import PluginManager


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
        unsatisfied = pipeline.inputs() - set(catalog.list())
        for ds_name in unsatisfied:
            catalog = catalog.shallow_copy()
            catalog.add(ds_name, self.create_default_data_set(ds_name))

        return super().run(pipeline, catalog, hook_manager, session_id)

    def create_default_data_set(self, ds_name: str) -> AbstractDataSet:
        return PickleDataSet(str(Path(self.data_paths[ds_name]) / f"{ds_name}.pickle"))
