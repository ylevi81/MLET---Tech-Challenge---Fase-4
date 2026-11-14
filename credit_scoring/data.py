"""Leitura do arquivo original da UCI e do CSV definido no contrato."""

from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from .contract import CUSTOMER_ID, SOURCE_FEATURES, TARGET


SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default%2Bof%2Bcredit%2Bcard%2Bclients.zip"
)
SOURCE_MEMBER = "default of credit card clients.xls"
SOURCE_COLUMNS = {
    "ID": CUSTOMER_ID,
    **SOURCE_FEATURES,
    "default payment next month": TARGET,
}


def download_reference(destination: Path) -> Path:
    """Mantém uma cópia local do XLS para repetir o treino sem rede."""
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary)
        archive_path = staging / "uci.zip"
        with urlopen(SOURCE_URL, timeout=60) as response, archive_path.open("wb") as output:
            copyfileobj(response, output)

        staged_xls = staging / SOURCE_MEMBER
        with ZipFile(archive_path) as archive:
            with archive.open(SOURCE_MEMBER) as source, staged_xls.open("wb") as output:
                copyfileobj(source, output)
        staged_xls.replace(destination)
    return destination


def read_batch(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".xls":
        frame = pd.read_excel(path, header=1, engine="xlrd")
        return frame.rename(columns=SOURCE_COLUMNS).loc[:, SOURCE_COLUMNS.values()]
    raise ValueError(f"Formato não suportado: {path.suffix}. Use CSV ou o XLS da UCI.")
