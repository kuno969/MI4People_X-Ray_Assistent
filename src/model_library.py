from abc import ABC, abstractmethod
import torchxrayvision as xrv
import torchvision
from torchvision.transforms.functional import normalize

class AbstractModelLibrary:
    def __init__(self):
        self.CHOICES = []
        self.LABELS = []

    @abstractmethod
    def get_model(self, choice: str):
        pass

    @abstractmethod
    def preprocess(self, img):
        pass

    @abstractmethod
    def get_target_layer(self, model):
        pass

class XRVModelLibrary(AbstractModelLibrary):
    def __init__(self):
        super().__init__()
        self.CHOICES = [
            "densenet121-res224-all",
            "densenet121-res224-rsna",
            "densenet121-res224-nih",
            "densenet121-res224-pc",
            "densenet121-res224-chex",
            "densenet121-res224-mimic_nb",
            "densenet121-res224-mimic_ch",
        ]

    def get_model(self, choice: str):
        model = xrv.models.DenseNet(weights=choice)
        self.LABELS = model.pathologies
        return model

    def get_target_layer(self, model):
        return model.features[-2][-1][-1]

    def preprocess(self, img):
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.CenterCrop(max(img.shape)),
                torchvision.transforms.Resize((224,224)),
                ])
        transformed_output = transform(img)
        normalized_output = normalize(transformed_output/255, 0, 1)
        return transformed_output, normalized_output
