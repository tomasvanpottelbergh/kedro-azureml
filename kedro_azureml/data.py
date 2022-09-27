import re
from pathlib import Path

import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml._artifacts._artifact_utilities import (
    download_artifact_from_aml_uri,
)
from azure.identity import DefaultAzureCredential
from kedro.extras.datasets.pandas import ParquetDataSet

"""
        >>> from pathlib import Path, PurePosixPath
        >>> import pandas as pd
        >>> from kedro.io import AbstractDataSet
        >>>
        >>>
        >>> class MyOwnDataSet(AbstractDataSet[pd.DataFrame, pd.DataFrame]):
        >>>     def __init__(self, filepath, param1, param2=True):
        >>>         self._filepath = PurePosixPath(filepath)
        >>>         self._param1 = param1
        >>>         self._param2 = param2
        >>>
        >>>     def _load(self) -> pd.DataFrame:
        >>>         return pd.read_csv(self._filepath)
        >>>
        >>>     def _save(self, df: pd.DataFrame) -> None:
        >>>         df.to_csv(str(self._filepath))
        >>>
        >>>     def _exists(self) -> bool:
        >>>         return Path(self._filepath.as_posix()).exists()
        >>>
        >>>     def _describe(self):
        >>>         return dict(param1=self._param1, param2=self._param2)
"""


class AzureMLParquetDataSet(ParquetDataSet):  # TODO: inherit from VersionedDataSet
    def __init__(self, filepath, name) -> None:
        super().__init__(filepath=filepath)  # TODO: add other kwargs
        self._name = name

    def _load(self) -> pd.DataFrame:
        # with _get_azureml_client(None, self._amlconfig) as ml_client:

        # TODO: read from credentials?
        # https://kedro.readthedocs.io/en/stable/data/data_catalog.html#feeding-in-credentials
        ml_client = MLClient.from_config(
            DefaultAzureCredential(), "./conf/base/aml_config.json"
        )

        data = ml_client.data.get(self._name, label="latest")
        # Remove workspaces part from path
        path = re.sub("workspaces/([^/]+)/", "", data.path)
        download_artifact_from_aml_uri(
            path, Path(self._filepath).parent, ml_client.datastores
        )

        # TODO: check whether file exists at path/rename
        # filename = path.split("/")[-1]

        return super()._load()
