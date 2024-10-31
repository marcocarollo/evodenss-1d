from __future__ import annotations

from argparse import ArgumentParser
import os
import logging
import random
import time
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch import nn

import evodenss
from evodenss.config.pydantic import Config, ConfigBuilder, DataSplits, get_config, get_fitness_extra_params
from evodenss.evolution import engine
from evodenss.evolution.grammar import Grammar
from evodenss.evolution.individual import Individual
from evodenss.misc.checkpoint import Checkpoint
from evodenss.misc.constants import DATASETS_INFO, DEFAULT_SEED, STATS_FOLDER_NAME, START_FROM_SCRATCH
from evodenss.misc.enums import DownstreamMode, FitnessMetricName, OptimiserType
from evodenss.misc.persistence import RestoreCheckpoint, build_overall_best_path
from evodenss.misc.utils import ConfigPairAction, is_valid_file, is_yaml_file
from evodenss.dataset.dataset_loader import DatasetProcessor, create_dataset_processor
from evodenss.networks.evaluators import BaseEvaluator
from evodenss.train.losses import MyCustomLoss
from evodenss.main import main


if TYPE_CHECKING:
    from evodenss.dataset.dataset_loader import DatasetType
    from torch.utils.data import Subset



#parent = Individual(grammar=Grammar(args.grammar_path, backup_path=get_config().checkpoints_path),
#                    network_architecture_config=get_config().network.architecture, 
#                    
#
#def get_loss_function():
#    config = get_config()
#    loss: str = config.network.learning.loss.type
#    if loss == "cross_entropy":
#        loss_function: nn.Module = torch.nn.CrossEntropyLoss()
#    elif loss == "argo":
#        loss_function: nn.Module = MyCustomLoss(config.network.learning.loss)
#    return loss_function
#
#evaluator = BaseEvaluator.create_evaluator(dataset_name="argo",
#                                            loss_function=get_loss_function(),
#                                            is_gpu_run=True)
#
#checkpoint  = Checkpoint(run=5,
#                         random_state=random.getstate(),
#                         numpy_random_state=np.random.get_state(),
#                         torch_random_state=torch.get_rng_state(),
#                         last_processed_generation=START_FROM_SCRATCH,
#                         total_epochs=0
#                         best_fitness=None,
#                         evaluator = evaluator,
#                         best_gen_ind_test_accuracy=0.0,
#                         population=None,
#                         parent=parent)

with open('/home/marco/Desktop/units/evodenss-1d/results/debug_argo/run_100/checkpoint.pkl', 'rb') as f:
    checkpoint = pkl.load(f)

if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config-path", '-c', required=True, help="Path to the config file to be used",
                        type=lambda x: is_yaml_file(parser, x))
    parser.add_argument("--dataset-name", '-d', required=True, help="Name of the dataset to be used",
                        type=str, choices=list(DATASETS_INFO.keys()))
    parser.add_argument("--grammar-path", '-g', required=True, help="Path to the grammar to be used",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--run", "-r", required=False, help="Identifies the run id and seed to be used",
                        type=int, default=0)
    parser.add_argument("--override", required=False, help="Sets or overrides values in the config file",
                        action=ConfigPairAction, nargs=2, metavar=('config_key','value'), default=[])
    parser.add_argument("--gpu-enabled", required=False, help="Runs the experiment in the GPU",
                        action='store_true')
    args: Any = parser.parse_args()

    start = time.time()
    torch.backends.cudnn.benchmark = True
    # loads config. it is a singleton
    # from now onwards you can access the config anywhere by calling
    # `get_config()` located in evodenss.config
    config: Config = ConfigBuilder(config_path=args.config_path,
                                   args_to_override=args.override)
    logger = setup_logger(config.checkpoints_path, args.run)
    os.makedirs(config.checkpoints_path, exist_ok=True)

    main(run=args.run,
         dataset_name=args.dataset_name,
         grammar=Grammar(args.grammar_path, backup_path=get_config().checkpoints_path),
         config=config,
         is_gpu_run=args.gpu_enabled,
         possible_checkpoint = checkpoint)

    end = time.time()
    time_elapsed = int(end - start)
    secs_elapsed = time_elapsed % 60
    mins_elapsed = time_elapsed//60 % 60
    hours_elapsed = time_elapsed//3600 % 60
    logger.info(f"Time taken to perform run: {compute_time_elapsed_human(time_elapsed)}")
    logging.shutdown()
