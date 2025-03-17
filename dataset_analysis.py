from __future__ import annotations

from argparse import ArgumentParser
import os
import logging
import random
import time
from typing import Any, Optional, TYPE_CHECKING
from torch.multiprocessing import Process

import numpy as np
import torch
from torch import nn

#import evodenss
#from evodenss.config.pydantic import Config, ConfigBuilder, DataSplits, get_config, get_fitness_extra_params
#from evodenss.evolution import engine
#from evodenss.evolution.grammar import Grammar
#from evodenss.misc.checkpoint import Checkpoint
#from evodenss.misc.constants import DATASETS_INFO, DEFAULT_SEED, STATS_FOLDER_NAME, START_FROM_SCRATCH
#from evodenss.misc.enums import DownstreamMode, FitnessMetricName, OptimiserType, Device
#from evodenss.misc.persistence import RestoreCheckpoint, build_overall_best_path
#from evodenss.misc.utils import ConfigPairAction, is_valid_file, is_yaml_file, plot_profiles
#from evodenss.dataset.dataset_loader import DatasetProcessor, create_dataset_processor, DatasetType
#from evodenss.networks.evaluators import BaseEvaluator
#from evodenss.train.losses import MyCustomLoss, MyCustomMSE
#
#if TYPE_CHECKING:
#    from evodenss.dataset.dataset_loader import DatasetType
#    from torch.utils.data import Subset
#
#logger: logging.Logger


test_data_chla = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/CHLA/test_data.pt')
test_data_bbp = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/BBP700/test_data.pt')
test_data_nitrate = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/NITRATE/test_data.pt')

#print max and min from [:,3,:]
print('these are the max and min values of CHLA:', test_data_chla[:,3,:].max(), test_data_chla[:,3,:].min())
print('these are the max and min values of BBP700:', test_data_bbp[:,3,:].max(), test_data_bbp[:,3,:].min())
print('these are the max and min values of NITRATE:', test_data_nitrate[:,3,:].max(), test_data_nitrate[:,3,:].min())



test_mean_chla = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/CHLA/test_mean.pt')
test_std_chla = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/CHLA/test_std.pt')

test_mean_bbp = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/BBP700/test_mean.pt')
test_std_bbp = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/BBP700/test_std.pt')

test_mean_nitrate = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/NITRATE/test_mean.pt')
test_std_nitrate = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/NITRATE/test_std.pt')

norm_chla = (test_data_chla - test_mean_chla) / (test_std_chla+1e-7)
norm_bbp = (test_data_bbp - test_mean_bbp) / (test_std_bbp+1e-7)
norm_nitrate = (test_data_nitrate - test_mean_nitrate) / (test_std_nitrate+1e-7)

n_data_chla = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/CHLA/n_test_data.pt')
n_data_bbp = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/BBP700/n_test_data.pt')
n_data_nitrate = torch.load('/u/mcarollo/evodenss-1d/evodenss/data/ds/NITRATE/n_test_data.pt')

# check if the two normalizations are the same, with a tolerance of 1e-7
print(torch.allclose(norm_chla, n_data_chla, atol=1e-7))
print(torch.allclose(norm_bbp, n_data_bbp, atol=1e-7))
print(torch.allclose(norm_nitrate, n_data_nitrate, atol=1e-7))

#now do the same denormalization
denorm_chla = n_data_chla * (test_std_chla+1e-7) + test_mean_chla
denorm_bbp = n_data_bbp * (test_std_bbp+1e-7) + test_mean_bbp
denorm_nitrate = n_data_nitrate * (test_std_nitrate+1e-7) + test_mean_nitrate

# check if the two denormalizations are the same, with a tolerance of 1e-7
print(torch.allclose(denorm_chla, test_data_chla, atol=1e-7))
print(torch.allclose(denorm_bbp, test_data_bbp, atol=1e-7))
print(torch.allclose(denorm_nitrate, test_data_nitrate, atol=1e-7))





