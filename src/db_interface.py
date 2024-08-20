import pandas as pd
import random

from azure.storage.blob import ContainerClient


def setup_container_client(account_key: str) -> ContainerClient:
    STORAGEACCOUNTURL = "https://computervision8800143538.blob.core.windows.net"
    STORAGEACCOUNTKEY = account_key
    CONTAINERNAME = "azureml-blobstore-cfd7f06d-0bfa-49bf-a8a6-12758a29e5dc"

    container_client = ContainerClient(
        account_url=STORAGEACCOUNTURL,
        container_name=CONTAINERNAME,
        credential=STORAGEACCOUNTKEY,
    )

    return container_client


def write_data_to_azure_blob(
    container_client: ContainerClient, blob_name: str, data: bytes
) -> None:
    container_client.upload_blob(blob_name, data, overwrite=True)


def read_data_from_azure_blob(
    container_client: ContainerClient, blob_name: str
) -> bytes:
    blob_data = container_client.download_blob(blob_name)

    return blob_data


def get_image_from_azure(container_client, image_filename, prefix="UI/05-12-2022_055200_UTC/images/"):
    blob_name =  prefix + image_filename

    return read_data_from_azure_blob(container_client, blob_name)


class MetadataStore:
    def __init__(self) -> None:
        self._df = None
        self._unique_labels = set()

    def read_from_azure(self, container_client: ContainerClient) -> None:
        LOCALFILENAME = "./meta.csv"

        with open(LOCALFILENAME, "wb") as my_blob:
            blob_data = read_data_from_azure_blob(
                container_client, "UI/05-11-2022_025810_UTC/Data_Entry_2017_v2020.csv"
            )
            blob_data.readinto(my_blob)

        self._df = pd.read_csv(LOCALFILENAME)

        for labels in self._df["Finding Labels"]:
            for label in labels.split("|"):
                self._unique_labels.add(label)

    def get_unique_labels(self) -> list[str]:
        return self._unique_labels

    def get_image_filenames(self, label: str) -> list[str]:
        return self._df[self._df["Finding Labels"].str.contains(label)][
            "Image Index"
        ].to_list()

    def get_random_image_filenames(self, n: int, label: str | None = None, inverse_label: bool = False) -> list[str]:
        if label is None:
            return random.sample(self._df["Image Index"].to_list(), n)
        else:
            if inverse_label:
                selector = ~self._df["Finding Labels"].str.contains(label)
            else:
                selector = self._df["Finding Labels"].str.contains(label)
            
            return random.sample(
                self._df[selector][
                    "Image Index"
                ].to_list(),
                n,
            )

    def get_full_label(self, image_filename: str) -> str:
        labels = self._df[self._df["Image Index"] == image_filename][
            "Finding Labels"
        ].to_list()[0]
        return ", ".join(labels.split("|"))
