from vilt.datasets import VizWizCaptionDataset
from .datamodule_base import BaseDataModule


class VizWizCaptionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VizWizCaptionDataset

    @property
    def dataset_cls_no_false(self):
        return VizWizCaptionDataset

    @property
    def dataset_name(self):
        return "vizwiz"
