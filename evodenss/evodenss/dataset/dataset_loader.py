from __future__ import annotations

from enum import unique, Enum
import importlib
import logging
from typing import Any, Optional, TypeAlias, TYPE_CHECKING
import types


import pandas as pd
import torch
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST

from evodenss.config.pydantic import AugmentationConfig, DataSplits, Labelled, Unlabelled
from evodenss.networks.transformers import LegacyTransformer, BarlowTwinsTransformer
from evodenss.misc.constants import DEFAULT_SEED


def from_string_to_tensor(string):
    threshold = 99999
    label = 1  # label equal to one means that the sample is good
    string = string[8:-2].split(",")
    out = torch.zeros(200)
    for ind in range(len(string)):
        if float(string[ind]) >= threshold:
            label = 0
        out[ind] = torch.tensor(float(string[ind]))
    return out, label


class FloatDataset(Dataset):

    def __init__(self, path_df=None, targets_path=None, transform=None):
        super().__init__()
        if path_df is not None:
            self.path_df = path_df
            self.df = pd.read_csv(self.path_df)
        if targets_path is not None:
            self.targets = torch.load(targets_path, weights_only=True)
        else:
            raise Exception("Paths should be given as input to initialize the Float class.")
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.df.iloc[0, :])

    def __getitem__(self, index):
        """Generates one sample of data"""
        try:
            self.samples = self.df.iloc[:, index + 1].tolist()  # Select sample
        except Exception as error:
            pass

        # self.samples = self.df.iloc[:, index + 1].tolist()  # Select sample

        year = torch.tensor(float(self.samples[0]))
        day_rad = torch.tensor(float(self.samples[1]))
        lat = torch.tensor(float(self.samples[2]))
        lon = torch.tensor(float(self.samples[3]))
        temp, label_temp = from_string_to_tensor(self.samples[4])
        psal, label_psal = from_string_to_tensor(self.samples[5])
        doxy, label_doxy = from_string_to_tensor(self.samples[6])

        label = label_doxy * label_psal * label_temp  # the label is equal to one only if I have data valid for all
        # depth
        
        return year, day_rad, lat, lon, temp, psal, doxy, self.targets[index]

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path_df=None, path_tensor = None, transform=None):
        super(MyDataset, self).__init__()
        if path_df is not None:
            self.path_df = path_df
            self.df = pd.read_csv(self.path_df)
            self.data = self.df.iloc[:, :-1].values
            self.targets = self.df.iloc[:, -1].values

        elif path_tensor is not None:
            self.df = torch.load(path_tensor, weights_only=True)
            self.data = self.df[:,:-1, :]
            self.targets = self.df[:,-1, :]
        else:
            raise Exception("Paths should be given as input to initialize the Float class.")
        self.transform = transform
     
        

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)
    

if TYPE_CHECKING:
    from torch.utils.data import Dataset

ConcreteDataset: TypeAlias = CIFAR10 | CIFAR100 | FashionMNIST | MNIST | FloatDataset |MyDataset

__all__ = ['DatasetType', 'DatasetProcessor']

logger = logging.getLogger(__name__)

def create_dataset_processor(augmentation_params: AugmentationConfig) -> DatasetProcessor:
    ssl_transformer = BarlowTwinsTransformer(augmentation_params.pretext)
    downstream_train_transformer = LegacyTransformer(augmentation_params.downstream)
    downstream_test_transformer = LegacyTransformer(augmentation_params.test)
    return DatasetProcessor(ssl_transformer, downstream_train_transformer, downstream_test_transformer)


def unlabel(y: Any) -> Tensor:
    return Tensor([-1])


@unique
class DatasetType(Enum):
    PRETEXT_TRAIN = "pretext_train"
    DOWNSTREAM_TRAIN = "downstream_train"
    VALIDATION = "validation"
    EVO_TEST = "evo_test"
    TEST = "test"


class DatasetProcessor:

    def __init__(self,
                 ssl_transformer: Optional[BarlowTwinsTransformer],
                 train_transformer: LegacyTransformer,
                 test_transformer: LegacyTransformer) -> None:
        self.ssl_transformer: Optional[BarlowTwinsTransformer] = ssl_transformer # pretext
        self.train_transformer: LegacyTransformer = train_transformer # downstream
        self.test_transformer: LegacyTransformer = test_transformer # downstream


    @classmethod
    def _split_sets(cls,
                    superset_idxs: NDArray[np.int_],
                    ratio: float,
                    stratify: NDArray[np.int_]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:

        assert ratio >= 0 and ratio <= 1

        subset_a_idxs: NDArray[np.int_]
        subset_b_idxs: NDArray[np.int_]
        if ratio == 1.0:
            subset_a_idxs = np.array([], dtype=np.int8)
            subset_b_idxs = superset_idxs
        elif ratio == 0.0:
            subset_a_idxs = superset_idxs
            subset_b_idxs = np.array([], dtype=np.int8)
        else:
            subset_a_idxs, subset_b_idxs = train_test_split(superset_idxs,
                                                            test_size=ratio,
                                                            shuffle=True,
                                                            stratify=None,
                                                            random_state=DEFAULT_SEED)
        return subset_a_idxs, subset_b_idxs


    def load_partitioned_dataset(self,
                                 dataset_name: str,
                                 proportions: DataSplits,
                                 seed: int) -> dict[DatasetType, Subset]:

        unlabelled_data: Optional[Dataset]
        train_labelled_data: Dataset
        evaluation_labelled_data: Dataset
        test_data: Dataset
        subset_idxs_dict: dict[DatasetType, np.ndarray] = {}
        subsets_dict: dict[DatasetType, Subset] = {}

        (unlabelled_data, train_labelled_data, evaluation_labelled_data, test_data) = \
            self._load_dataset(dataset_name)

        labelled_data_settings: Labelled = proportions.labelled
        unlabelled_data_settings: Optional[Unlabelled] = proportions.unlabelled

        # the targets are the same for the datasets that derive from the original training set
        # (except for the unlabelled, whose labels have been removed)
        targets: NDArray[np.int_] = np.array(train_labelled_data.targets)

        labelled_data_ratio: float = labelled_data_settings.percentage/100
        dataset_idxs: NDArray[np.int_] = np.arange(0, len(train_labelled_data))

        unlabelled_idx: NDArray[np.int_]
        labelled_idx: NDArray[np.int_]
        unlabelled_idx, labelled_idx = self._split_sets(dataset_idxs, labelled_data_ratio, targets)
        
        train_labelled_idx: NDArray[np.int_]
        evo_test_idx: NDArray[np.int_]
        test_partition_value: float = labelled_data_settings.evo_test.partition_ratio
        test_ratio: float = test_partition_value / labelled_data_ratio
        train_labelled_idx, evo_test_idx = self._split_sets(labelled_idx, test_ratio, targets[labelled_idx])
        subset_idxs_dict[DatasetType.EVO_TEST] = evo_test_idx

        validation_idx: NDArray[np.int_]
        val_partition_value: float = labelled_data_settings.validation.partition_ratio
        validation_ratio: float = val_partition_value / (labelled_data_ratio - test_partition_value)
        train_labelled_idx, validation_idx = self._split_sets(train_labelled_idx,
                                                              validation_ratio,
                                                              targets[train_labelled_idx])
        subset_idxs_dict[DatasetType.VALIDATION] = validation_idx
        subset_idxs_dict[DatasetType.DOWNSTREAM_TRAIN] = train_labelled_idx
        subset_idxs_dict[DatasetType.EVO_TEST] = evo_test_idx
        if unlabelled_data_settings is not None:
            # we assume that the training subsets from downstream and pretext might contain overlapping data
            subset_idxs_dict[DatasetType.PRETEXT_TRAIN] = np.concatenate((unlabelled_idx, train_labelled_idx))
            # subset_idxs_dict[DatasetType.PRETEXT_TRAIN] = unlabelled_idx

        sample_size: int|float
        settings_to_use: dict[str, Any]
        for dataset_type, idxs in subset_idxs_dict.items():
            dataset_to_use: ConcreteDataset
            if dataset_type == DatasetType.PRETEXT_TRAIN and unlabelled_data_settings is not None:
                dataset_to_use = unlabelled_data
                settings_to_use = unlabelled_data_settings.__dict__
            else:
                dataset_to_use = \
                    evaluation_labelled_data if dataset_type == DatasetType.EVO_TEST else train_labelled_data
                settings_to_use = labelled_data_settings.__dict__

            if len(idxs) == 0:
                subsets_dict[dataset_type] = Subset(dataset_to_use, [])
                continue

            sample_size = settings_to_use[dataset_type.value].amount_to_use
            sample_size = int(sample_size * len(idxs)) if isinstance(sample_size, float) else sample_size
            # if `replacement` is true then we vary the seed via function parameter
            seed_to_use: int = seed if settings_to_use[dataset_type.value].replacement is True else DEFAULT_SEED
            idxs_to_use: NDArray[np.int_]
            if sample_size >= len(idxs):
                if sample_size > len(idxs):
                    logger.warning(f"Partition {dataset_type} has size {len(idxs)}, "
                                   f"but {sample_size} samples are being requested")
                sample_size = len(idxs)
                idxs_to_use = idxs
            else:
                _, idxs_to_use = train_test_split(idxs,
                                                       test_size=sample_size,
                                                       shuffle=True,
                                                       stratify=targets[idxs],
                                                       random_state=seed_to_use)
            subsets_dict[dataset_type] = Subset(dataset_to_use, idxs_to_use.tolist())
        subsets_dict[DatasetType.TEST] = Subset(test_data, list(range(len(test_data.targets)))) # type: ignore
        return subsets_dict


    def _load_dataset(self, dataset_name: str) -> tuple[ConcreteDataset,
                                                        ConcreteDataset,
                                                        ConcreteDataset,
                                                        ConcreteDataset]:
        # Check if it's a dataset from torchvision
        module_obj: types.ModuleType = importlib.import_module("torchvision.datasets")
        class_names = [x.lower() for x in dir(module_obj)]
        dataset_name = dataset_name.replace('-', '')

        if dataset_name in class_names:
            # If found in torchvision datasets
            class_name_index: int = class_names.index(dataset_name)
            dataset_class = getattr(module_obj, dir(module_obj)[class_name_index])
        else:
            # If not found, fallback to custom dataset
            dataset_class = FloatDataset

        dataset_class: type[ConcreteDataset] = dataset_class

        if dataset_class == FloatDataset:

            unlabelled_data = None
            train_labelled_data = dataset_class(path_df='/u/dssc/mcarol00/main/evodenss-1d/evodenss/data/ds/CHLA/float_ds_sf_train.csv',
                                                targets_path='/u/dssc/mcarol00/main/evodenss-1d/evodenss/data/ds/CHLA/targets_train.pt')
            evaluation_labelled_data = dataset_class(path_df='/u/dssc/mcarol00/main/evodenss-1d/evodenss/data/ds/CHLA/float_ds_sf_train.csv',
                                                targets_path='/u/dssc/mcarol00/main/evodenss-1d/evodenss/data/ds/CHLA/targets_train.pt') #qui era val data ma casino
            test_data = dataset_class(path_df='/u/dssc/mcarol00/main/evodenss-1d/evodenss/data/ds/CHLA/float_ds_sf_test.csv',
                                      targets_path='/u/dssc/mcarol00/main/evodenss-1d/evodenss/data/ds/CHLA/targets_test.pt') #/home/marco/Desktop/units/evodenss-1d/evodenss/data/ds/BBP700/test_data.pt
        else:
            unlabelled_data = dataset_class(
                root="data",
                train=True,
                download=True,
                transform=self.ssl_transformer,
                target_transform = unlabel
            )
            train_labelled_data = dataset_class(
                root="data",
                train=True,
                download=True,
                transform=self.train_transformer
            )
            evaluation_labelled_data = dataset_class(
                root="data",
                train=True,
                download=True,
                transform=self.test_transformer
            )
            test_data = dataset_class(
                root="data",
                train=False,
                download=True,
                transform=self.test_transformer
            )
            print(f"length of train_labelled_data: {len(train_labelled_data)}")
            print(f"length of evaluation_labelled_data: {len(evaluation_labelled_data)}")
            print(f"length of test_data: {len(test_data)}")
            
        return unlabelled_data, train_labelled_data, evaluation_labelled_data, test_data


    @classmethod
    def get_data_loaders(cls,
                         dataset: dict[DatasetType, Subset],
                         partitions_to_get: list[DatasetType],
                         batch_size: int) -> dict[DatasetType, DataLoader]:
        is_drop_last: bool
        loaders_dict: dict[DatasetType, DataLoader] = {}
        g = Generator()
        for p in partitions_to_get:
            g.manual_seed(DEFAULT_SEED)
            #during bt training if the the last batch has 1 element, training breaks at last batch norm.
            #therefore, we drop the last batch
            is_drop_last = True if p == DatasetType.PRETEXT_TRAIN else False
            loaders_dict[p] = DataLoader(dataset[p],
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=16,
                                         drop_last=is_drop_last,
                                         pin_memory=True,
                                         generator=g)
        return loaders_dict


