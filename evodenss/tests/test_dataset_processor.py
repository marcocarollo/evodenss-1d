import unittest

import numpy as np
from parameterized import parameterized
import sys
sys.path.append('..')

from evodenss.config.pydantic import DataSplits, DownstreamTrain, EvoTest, Labelled, Validation, init_context
from evodenss.dataset.dataset_loader import DatasetProcessor, DatasetType
from evodenss.misc.constants import DEFAULT_SEED
from evodenss.misc.enums import LearningType
from evodenss.networks.transformers import LegacyTransformer


class Test(unittest.TestCase):

    @parameterized.expand([
        (0.0, np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]), np.array([])),
        (0.25, np.array([14,17,0,2,12,19,11,9,8,1,16,18,5,6,3]), np.array([7,10,4,15,13])),
        (0.5, np.array([14,19,2,11,3,18,12,5,9,1]), np.array([0,17,4,13,6,8,7,16,10,15])),
        (1.0, np.array([]), np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]))
    ])
    def test_split_sets(self, ratio, expected_set_a, expected_set_b):
        dataset_ids = np.arange(0, 20)
        fake_labels = np.concatenate([np.ones(10, dtype=np.int8), np.zeros(10, dtype=np.int8)])
        set_a, set_b = DatasetProcessor._split_sets(
            dataset_ids,
            ratio=ratio,
            stratify=fake_labels
        )
        np.testing.assert_array_equal(set_a, expected_set_a)
        np.testing.assert_array_equal(set_b, expected_set_b)

    def test_load_partitioned_dataset(self):
        dataset_processor = DatasetProcessor(
            ssl_transformer=None,
            train_transformer=LegacyTransformer({}),
            test_transformer=LegacyTransformer({})
        )
        with init_context({'learning_type': LearningType.supervised}):
            train_partition_ratio = 0.007
            validation_partition_ratio = 0.001
            test_partition_ratio = 0.002
            data_splits = DataSplits(
                unlabelled=None,
                labelled=Labelled(
                    percentage=1,
                    downstream_train=DownstreamTrain(
                        partition_ratio=train_partition_ratio,
                        amount_to_use=1.0,
                        replacement=False),
                    validation=Validation(
                        partition_ratio=validation_partition_ratio,
                        amount_to_use=1.0,
                        replacement=False),
                    evo_test=EvoTest(
                        partition_ratio=test_partition_ratio,
                        amount_to_use=1.0,
                        replacement=False)
                )
            )
        subset_dict = dataset_processor.load_partitioned_dataset(
                dataset_name="mnist",
                proportions=data_splits,
                seed=DEFAULT_SEED)
        print(len(subset_dict[DatasetType.DOWNSTREAM_TRAIN]), len(subset_dict[DatasetType.VALIDATION]), len(subset_dict[DatasetType.EVO_TEST]))

        self.assertEqual(len(subset_dict[DatasetType.DOWNSTREAM_TRAIN].indices)/60000,
                         train_partition_ratio)
        self.assertEqual(len(subset_dict[DatasetType.VALIDATION].indices)/60000,
                         validation_partition_ratio)
        self.assertEqual(len(subset_dict[DatasetType.EVO_TEST].indices)/60000,
                         test_partition_ratio)


if __name__ == '__main__':
    unittest.main()
