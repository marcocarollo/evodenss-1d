from argparse import _AppendAction
import os
from typing import Any, NamedTuple, NewType, no_type_check
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from argparse import ArgumentParser
from evodenss.misc.constants import INTERVALS, MAX_PRESSURES

InputLayerId = NewType('InputLayerId', int)
LayerId = NewType('LayerId', int)
#PunctualLayerId = NewType('PunctualLayerId', int)
#PunctualInputId = NewType('PunctualInputId', int)

def is_valid_file(parser: ArgumentParser, arg: Any) -> object:
    if not os.path.isfile(arg):
        parser.error(f"The file {arg} does not exist!")
    else:
        return arg

def is_yaml_file(parser: ArgumentParser, arg: Any) -> object:
    if is_valid_file(parser, arg):
        if not arg.endswith(".yaml"):
            parser.error(f"The file {arg} is not a yaml file")
        else:
            return arg
    parser.error(f"The file {arg} is not a yaml file")


class ConfigPair(NamedTuple):
    key: str
    value: Any


class ConfigPairAction(_AppendAction):

    @no_type_check
    def _copy_items(items):
        if items is None:
            return []
        # The copy module is used only in the 'append' and 'append_const'
        # actions, and it is needed only when the default value isn't a list.
        # Delay its import for speeding up the common case.
        if isinstance(items, list):
            return items[:]
        import copy
        return copy.copy(items)

    @no_type_check
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        if nargs != 2:
            raise ValueError(f"ConfigPairAction requires two args per flag. Current nargs = {nargs}")
        super(_AppendAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    @no_type_check
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = ConfigPairAction._copy_items(items)
        items.append(ConfigPair(*values))
        setattr(namespace, self.dest, items)


class InvalidNetwork(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

def plot_profiles(ds: DataLoader, dir: str, variable:str, gen: int, device: torch.device):

    
    dir_profile = dir + "/profile" + f"/generation_{gen}"
    
    if not os.path.exists(dir_profile):
        os.makedirs(dir_profile, exist_ok=True)

    
    #dp_rate = info_df['dp_rate'].item() aggiungere la dropout rate

    max_pres = MAX_PRESSURES[variable]
    interval = INTERVALS[variable]

    # Path of the saved models
    path_model = dir +  "/overall_best/model.pt"
    path_weights = dir +  "/overall_best/weights.pt"

    # Upload and evaluate all the models necessary
    model = torch.load(path_model)
    model.load_state_dict(torch.load(path_weights))
    model.eval()       
        
    for year, day_rad, lat, lon, temp, psal, doxy, output_variable in ds:
        year, day_rad, lat, lon, temp, psal, doxy, target = year.to(device.value, non_blocking=True), \
            day_rad.to(device.value, non_blocking=True), \
            lat.to(device.value, non_blocking=True), \
            lon.to(device.value, non_blocking=True), \
            temp.to(device.value, non_blocking=True), \
            psal.to(device.value, non_blocking=True), \
            doxy.to(device.value, non_blocking=True), \
            output_variable.to(device.value, non_blocking=True)
        data_conv = torch.stack([temp, psal, doxy], dim=1)
        inputs = tuple([year, day_rad, lat, lon, data_conv])
    
        output_test = model(inputs).unsqueeze(1) #forse c'è da fare un unsqueeze
        output_variable = output_variable.unsqueeze(1)
        
        
        
        depth_output = np.linspace(0, max_pres, len(output_test[0, 0, :].detach().cpu().numpy()))
        depth_variable = np.linspace(0, max_pres, len(output_variable[0, 0, :].detach().cpu().numpy()))

        if variable == "BBP700":
            output_variable = output_variable / 1000
            output_test = output_test / 1000

        plt.plot(output_test[0, 0, :].detach().cpu().numpy(), depth_output, label=f"generated {variable}")
        plt.plot(output_variable[0, 0, :].detach().cpu().numpy(), depth_variable, label=f"measured {variable}")
        plt.gca().invert_yaxis()

        plt.legend()

        plt.savefig(dir_profile + f"/profile_{year.item()}_{day_rad.item()}_{round(lat.item(), 2)}_{round(lon.item(), 2)}.png")
        # plt.show()
        plt.close()
    
