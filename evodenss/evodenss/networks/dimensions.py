from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import TYPE_CHECKING

from evodenss.misc.enums import LayerType


if TYPE_CHECKING:
    from evodenss.networks.phenotype_parser import Layer

@dataclass
class Dimensions1d:
    channels: int
    length: int

    @classmethod
    def from_layer(cls, layer: Layer, input_dimensions: "Dimensions1d") -> "Dimensions1d":
        out_channels: int
        length: int
        kernel_size: int
        padding: int
        if layer.layer_type == LayerType.CONV1D:
            out_channels = layer.layer_parameters['out_channels']
            if layer.layer_parameters['padding'] == "same":
                length = input_dimensions.length
            elif layer.layer_parameters['padding'] == "valid":
                kernel_size = layer.layer_parameters['kernel_size']
                length = floor((input_dimensions.length - kernel_size) / layer.layer_parameters['stride']) +1
            elif isinstance(layer.layer_parameters['padding'], int):
                padding = layer.layer_parameters['padding']
                kernel_size = layer.layer_parameters['kernel_size']
                length = floor((input_dimensions.length - kernel_size + padding * 2) / layer.layer_parameters['stride']) +1
            return cls(out_channels, length)
        elif layer.layer_type == LayerType.DECONV1D:
            out_channels = layer.layer_parameters['out_channels']
            padding = int(layer.layer_parameters['padding'])
            kernel_size = layer.layer_parameters['kernel_size']
            length = (input_dimensions.length - 1) * layer.layer_parameters['stride'] + kernel_size - 2 * padding
            return cls(out_channels, length)
        #elif layer.layer_type in [LayerType.POOL1D_AVG, LayerType.POOL1D_MAX]:
        #    assert isinstance(layer.layer_parameters['padding'], str) is True
        #    out_channels = input_dimensions.channels
        #    kernel_size = layer.layer_parameters['kernel_size']
        #    if layer.layer_parameters['padding'] == "valid":
        #        padding = 0
        #    elif layer.layer_parameters['padding'] == "same":
        #        paddings: tuple[int, int] = \
        #            input_dimensions.compute_adjusting_padding(kernel_size, layer.layer_parameters['stride'])
        #        padding = paddings[0] + paddings[1]
        #    length = ceil((input_dimensions.length - kernel_size + 1) / layer.layer_parameters['stride']) + padding
        #    return cls(out_channels, length)
        elif layer.layer_type in [LayerType.BATCH_NORM, LayerType.BATCH_NORM1D, LayerType.DROPOUT, LayerType.IDENTITY, LayerType.RELU_AGG]:
            return input_dimensions
        #elif layer.layer_type == LayerType.BATCH_NORM_PROJ1D:
            return cls(input_dimensions.flatten(), length=1)
        elif layer.layer_type == LayerType.FC:
            return cls(layer.layer_parameters['out_features'], length=1)
        elif layer.layer_type == LayerType.PUNCTUAL_MLP:
            return Dimensions1d(1,200)
        else:
            raise ValueError(f"Can't create Dimensions1d object for layer [{layer.layer_type}]")
    def flatten(self) -> int:
        return self.channels * self.length
    
    def compute_adjusting_padding(self,
                                  kernel_size: int,
                                  stride: int) -> tuple[int, int]:
        padding: float = (self.length - (self.length - kernel_size + 1) / stride) / 2
        padding_left: int = ceil(padding)
        padding_right: int = floor(padding)
        return (padding_left, padding_right)

    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.channels, self.length))

    #let's overwrite the add method to add two dimensions1d objects so that i get a new object with the sum of the two lengths
    #we need to assert that the two objects have the same length

    def __add__(self, other: Dimensions1d) -> Dimensions1d:
        assert self.length == other.length
        return Dimensions1d(self.channels + other.channels, self.length)


    

@dataclass
class Dimensions:
    channels: int
    height: int
    width: int

    @classmethod
    def from_layer(cls, layer: Layer, input_dimensions: "Dimensions") -> "Dimensions":
        out_channels: int
        height: int
        width: int
        kernel_size: int
        padding_w: int
        padding_h: int
        if layer.layer_type == LayerType.CONV:
            out_channels = layer.layer_parameters['out_channels']
            if layer.layer_parameters['padding'] == "same":
                height = input_dimensions.height
                width = input_dimensions.width
            elif layer.layer_parameters['padding'] == "valid":
                kernel_size = layer.layer_parameters['kernel_size']
                height = ceil((input_dimensions.height - kernel_size + 1) / layer.layer_parameters['stride'])
                width = ceil((input_dimensions.width - kernel_size + 1) / layer.layer_parameters['stride'])
            elif isinstance(layer.layer_parameters['padding'], tuple):
                padding_h = layer.layer_parameters['padding'][0]
                padding_w = layer.layer_parameters['padding'][1]
                kernel_size_h: int = layer.layer_parameters['kernel_size'][0]
                kernel_size_w: int = layer.layer_parameters['kernel_size'][1]
                height = ceil((input_dimensions.height - kernel_size_h + 1) / \
                              layer.layer_parameters['stride']) + padding_h * 2
                width = ceil((input_dimensions.width - kernel_size_w + 1) / \
                             layer.layer_parameters['stride']) + padding_w * 2
            return cls(out_channels, height, width)
        elif layer.layer_type in [LayerType.POOL_AVG, LayerType.POOL_MAX]:
            assert isinstance(layer.layer_parameters['padding'], str) is True
            out_channels = input_dimensions.channels
            kernel_size = layer.layer_parameters['kernel_size']
            if layer.layer_parameters['padding'] == "valid":
                padding_w = padding_h = 0
            elif layer.layer_parameters['padding'] == "same":
                paddings: tuple[int, int, int, int] = \
                    input_dimensions.compute_adjusting_padding(layer.layer_parameters['kernel_size'],
                                                               layer.layer_parameters['stride'])
                padding_w = paddings[2] + paddings[3]
                padding_h = paddings[0] + paddings[1]
            kernel_w: int
            kernel_h: int
            if isinstance(kernel_size, int):
                kernel_w = kernel_h = kernel_size
            elif isinstance(kernel_size, tuple):
                kernel_h = kernel_size[0]
                kernel_w = kernel_size[1]
            height = ceil((input_dimensions.height - kernel_h + 1) / layer.layer_parameters['stride']) + padding_h
            width = ceil((input_dimensions.width - kernel_w + 1) / layer.layer_parameters['stride']) + padding_w
            return cls(out_channels, height, width)
        elif layer.layer_type in [LayerType.BATCH_NORM, LayerType.DROPOUT, LayerType.IDENTITY, LayerType.RELU_AGG]:
            return input_dimensions
        elif layer.layer_type == LayerType.BATCH_NORM_PROJ:
            return cls(input_dimensions.flatten(), height=1, width=1)
        elif layer.layer_type == LayerType.FC:
            return cls(layer.layer_parameters['out_features'], height=1, width=1)
        else:
            raise ValueError(f"Can't create Dimensions object for layer [{layer.layer_type}]")


    def compute_adjusting_padding(self,
                                  kernel_size: int,
                                  stride: int) -> tuple[int, int, int, int]:
        padding_w: float = (self.width - (self.width - kernel_size + 1) / stride) / 2
        padding_h: float = (self.height - (self.height - kernel_size + 1) / stride) / 2
        padding_left: int = ceil(padding_w)
        padding_right: int = floor(padding_w)
        padding_top: int = ceil(padding_h)
        padding_bottom: int = floor(padding_h)
        return (padding_left, padding_right, padding_top, padding_bottom)

    def flatten(self) -> int:
        return self.channels * self.height * self.width

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.channels, self.height, self.width))
