from abc import ABC, abstractmethod
import torchxrayvision as xrv
import torchvision
from torchvision.transforms.functional import normalize

class AbstractModelLibrary:
    def __init__(self):
        self.CHOICES = []
        self.LABELS = []
        self.TARGET_LAYERS = {}

    @abstractmethod
    def get_model(self, choice: str):
        pass

    @abstractmethod
    def preprocess(self, img):
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
        self.LABELS = model.targets
        self.TARGET_LAYERS["layer+4-1+2"]=model.features[4][-1][2]
        self.TARGET_LAYERS["layer+4-1+5"]=model.features[4][-1][5]
        self.TARGET_LAYERS["layer+6-1+2"]=model.features[6][-1][2]
        self.TARGET_LAYERS["layer+6-1+5"]=model.features[6][-1][2]
        self.TARGET_LAYERS["layer+8-1+2"]=model.features[8][-1][2]
        self.TARGET_LAYERS["layer+8-1+5"]=model.features[8][-1][2]
        self.TARGET_LAYERS["layer+10-1+2"]=model.features[10][-1][2]
        self.TARGET_LAYERS["layer+10-1+5"]=model.features[10][-1][2]
        return model

    def preprocess(self, img):
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.CenterCrop(max(img.shape)),
                torchvision.transforms.Resize((224,224)),
                ])
        transformed_output = transform(img)
        normalized_output = normalize(transformed_output, 0, 1)
        rescaled_output = 1024*normalized_output
        return transformed_output, rescaled_output
