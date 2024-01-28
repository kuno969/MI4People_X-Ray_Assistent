import pandas as pd
from azureml.fsspec import AzureMachineLearningFileSystem

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureSasCredential

import pandas as pd

import base64

def read_data_from_azure_blob(account_key, blob_name):
    STORAGEACCOUNTURL= "https://computervision8800143538.blob.core.windows.net"
    STORAGEACCOUNTKEY = account_key
    CONTAINERNAME = "azureml-blobstore-cfd7f06d-0bfa-49bf-a8a6-12758a29e5dc"
    BLOBNAME = blob_name

    #download from blob
    blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
    blob_client_instance = blob_service_client_instance.get_blob_client(CONTAINERNAME, BLOBNAME, snapshot=None)
    blob_data = blob_client_instance.download_blob()

    return blob_data

class MetadataStore:
    def __init__(self):
        self._df = None
        self._unique_labels = set()

    def read_from_azure(self, account_key):
        LOCALFILENAME = "./meta.csv"
        
        with open(LOCALFILENAME, "wb") as my_blob:
            blob_data = read_data_from_azure_blob(account_key,"UI/05-11-2022_025810_UTC/Data_Entry_2017_v2020.csv")
            blob_data.readinto(my_blob)

        self._df = pd.read_csv(LOCALFILENAME)

        for labels in self._df["Finding Labels"]:
            for label in labels.split("|"):
                self._unique_labels.add(label)

    def get_unique_labels(self):
        return self._unique_labels
    
    def get_image_filenames(self, label):
        return self._df[self._df["Finding Labels"].str.contains(label)]["Image Index"].to_list()
    
    def get_full_label(self, image_filename):
        return self._df[self._df["Image Index"]==image_filename]["Finding Labels"].to_list()[0]

def get_image_from_azure(account_key, image_filename):
    blob_name = "UI/05-12-2022_055200_UTC/images/" + image_filename
    
    return read_data_from_azure_blob(account_key,blob_name)