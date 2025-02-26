from typing import Any

DATASETS_INFO: dict[str, dict[str, Any]] = {
    "mnist": {
        "expected_input_dimensions": (1, 32, 32),
        "classes": 10
    },
    "fashion-mnist": {
        "expected_input_dimensions": (1, 32, 32),
        "classes": 10
    },
    "cifar10": {
        "expected_input_dimensions": (3, 32, 32),
        "classes": 10
    },
    "cifar100": {
        "expected_input_dimensions": (3, 32, 32),
        "classes": 100
    },
     "argo": {
        "expected_input_dimensions": (1,200),
    }

}
#, "svhn", "cifar10",
# "cifar100-fine", "cifar100-coarse", "tiny-imagenet"]
#INPUT_DIMENSIONS: tuple[int, int, int] = (1, 32, 32)

OVERALL_BEST_FOLDER = "overall_best"
STATS_FOLDER_NAME = "statistics"
CHANNEL_INDEX = 1
MODEL_FILENAME = "model.pt"
WEIGHTS_FILENAME = "weights.pt"
METADATA_FILENAME = "metadata"
SEPARATOR_CHAR = "-"
START_FROM_SCRATCH = -1  #-1 perché poi quando iteriamo su tutte le generations facciamo +1, quindi parte da 0 fino al tot numero di generations
DEFAULT_SEED = 0

MAX_PRESSURES = {"NITRATE": 1000,
                     "CHLA": 200,
                     "BBP700": 200}

INTERVALS = {"NITRATE": 5,
                 "CHLA": 1,
                 "BBP700": 1}