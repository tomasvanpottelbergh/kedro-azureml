import inspect
import json
import re
from abc import ABCMeta
from functools import partial
from operator import attrgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Type, Union

import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml._artifacts._artifact_utilities import (
    download_artifact_from_aml_uri,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from cachetools import cachedmethod
from cachetools.keys import hashkey
from kedro.io.core import (
    AbstractVersionedDataSet,
    DataSetNotFoundError,
    Version,
    VersionNotFoundError,
    get_protocol_and_path,
    parse_dataset_definition,
)


class DynamicInheritance(ABCMeta):
    def __call__(cls, supertype, *args, **kwargs):
        """Dynamically set the the superclass based on the supertype argument."""
        if isinstance(supertype, str):
            # Resolve supertype string to type
            supertype, _ = parse_dataset_definition({"type": supertype})
        elif not isinstance(supertype, type):
            raise TypeError("Parameter 'supertype' must be a string or a type")

        new_cls = type(cls.__name__, (cls, supertype), {})

        return super(DynamicInheritance, new_cls).__call__(supertype, *args, **kwargs)

    def __new__(mcls, name, bases, namespace, /, **kwargs):
        """Fix the __module__ attribute being set to abc for classes using metaclasses
        inheriting from ABCMeta. Source: https://bugs.python.org/issue28869
        """
        if "__module__" not in namespace:
            # globals()['__name__'] gives 'abc'
            frame = inspect.currentframe()
            if frame is not None:
                # IronPython?
                caller_globals = frame.f_back.f_globals
                namespace["__module__"] = caller_globals["__name__"]
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        return cls


class AzureMLInputDataSet(AbstractVersionedDataSet, metaclass=DynamicInheritance):
    def __init__(
        self,
        supertype: Union[str, Type],
        filepath: str,
        name: str,
        credentials: Dict[str, Any],
        version: Version = None,
        **kwargs,
    ) -> None:

        protocol, _ = get_protocol_and_path(filepath, version)
        if protocol != "file":
            raise ValueError("Filepath can only be local path")

        self._name = name
        self._supertype = supertype

        # TODO: support other authentication methods
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(credentials))
            self._ml_client = MLClient.from_config(
                credential=DefaultAzureCredential(), path=str(config_path.absolute())
            )

        if version is None:
            version = Version(self._get_latest_version(), None)

        super().__init__(filepath=filepath, version=version, **kwargs)

    def _get_latest_version(self) -> str:
        try:
            return self._ml_client.data.get(self._name, label="latest").version
        except ResourceNotFoundError:
            raise DataSetNotFoundError(f"Did not find Azure ML Data Asset for {self}")

    @cachedmethod(cache=attrgetter("_version_cache"), key=partial(hashkey, "load"))
    def _fetch_latest_load_version(self) -> str:
        return self._get_latest_version()

    def _get_aml_dataset(self):
        return self._ml_client.data.get(self._name, version=self.resolve_load_version())

    def _load(self) -> pd.DataFrame:
        try:
            data = self._get_aml_dataset()
        except ResourceNotFoundError:
            raise VersionNotFoundError(
                f"Did not find version {self.resolve_load_version()} for {self}"
            )

        # Convert to short URI format to avoid problems
        path = re.sub("workspaces/([^/]+)/", "", data.path)
        path = re.sub("subscriptions/([^/]+)/", "", path)
        path = re.sub("resource[gG]roups/([^/]+)/", "", path)

        download_artifact_from_aml_uri(
            path, self._get_load_path().parent, self._ml_client.datastores
        )

        # TODO: check whether file exists at path/rename
        # filename = path.split("/")[-1]

        return super()._load()

    def _save(self, data: pd.DataFrame) -> None:
        raise NotImplementedError("Saving to a AzureMLInputDataSet is not supported")

    def _exists(self) -> bool:
        try:
            self._get_aml_dataset()
        except ResourceNotFoundError:
            return False

        return True

    def _describe(self) -> Dict[str, Any]:
        return dict(name=self._name, **super()._describe())

    def convert_to_supertype(self):
        self.__class__ = self._supertype
