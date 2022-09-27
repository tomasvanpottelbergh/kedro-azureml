import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml._artifacts._artifact_utilities import (
    download_artifact_from_aml_uri,
)
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from kedro.extras.datasets.pandas import ParquetDataSet
from kedro.io.core import Version, get_protocol_and_path

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


class AzureMLParquetDataSet(ParquetDataSet):  # TODO: inherit dynamically? make mixin?
    def __init__(
        self,
        filepath: str,
        name: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ) -> None:

        protocol, _ = get_protocol_and_path(filepath, version)
        if protocol != "file":
            raise ValueError("Filepath can only be local path")

        super().__init__(
            filepath=filepath,
            load_args=load_args,
            save_args=save_args,
            version=None,  # Do not apply versioning to local files
            credentials=credentials,
            fs_args=fs_args,
        )

        self._name = name
        self.__version = version

        # TODO: read from credentials?
        # https://kedro.readthedocs.io/en/stable/data/data_catalog.html#feeding-in-credentials
        self._ml_client = MLClient.from_config(
            DefaultAzureCredential(), "./conf/base/aml_config.json"
        )

    def _load(self) -> pd.DataFrame:
        # with _get_azureml_client(None, self._amlconfig) as ml_client:
        if self.__version and self.__version.load:
            version, label = self.__version.load, None
        else:
            version, label = None, "latest"

        data = self._ml_client.data.get(self._name, version=version, label=label)

        # Convert to short URI format to avoid problems
        path = re.sub("workspaces/([^/]+)/", "", data.path)
        path = re.sub("subscriptions/([^/]+)/", "", path)
        path = re.sub("resource[gG]roups/([^/]+)/", "", path)

        download_artifact_from_aml_uri(
            path, Path(self._filepath).parent, self._ml_client.datastores
        )

        # TODO: check whether file exists at path/rename
        # filename = path.split("/")[-1]

        return super()._load()

    def _save(self, data: pd.DataFrame) -> None:
        super()._save(data)

        data_asset = Data(
            path=self._filepath,
            type=AssetTypes.URI_FILE,
            description="Data asset registered by the kedro-azureml plugin",
            name=self._name,
        )

        self._ml_client.data.create_or_update(data_asset)

    def _exists(self) -> bool:
        try:
            self._ml_client.data.get(self._name, label="latest")
        except ResourceNotFoundError:
            return False

        return True

    def _describe(self) -> Dict[str, Any]:
        return dict(name=self._name, **super()._describe())
