# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Sequence

from antgo.framework.helper.parallel.utils import is_module_wrapper
from antgo.framework.helper.runner import HOOKS, Hook, rsetattr, rgetattr
from antgo.framework.helper.utils import build_from_cfg


@HOOKS.register_module()
class RTMOModeSwitchHook(Hook):
    """A hook to switch the mode of RTMO during training.

    This hook allows for dynamic adjustments of model attributes at specified
    training epochs. It is designed to modify configurations such as turning
    off specific augmentations or changing loss functions at different stages
    of the training process.

    Args:
        epoch_attributes (Dict[str, Dict]): A dictionary where keys are epoch
        numbers and values are attribute modification dictionaries. Each
        dictionary specifies the attribute to modify and its new value.

    Example:
        epoch_attributes = {
            5: [{"attr1.subattr": new_value1}, {"attr2.subattr": new_value2}],
            10: [{"attr3.subattr": new_value3}]
        }
    """

    def __init__(self, epoch_attributes: Dict[int, Dict]):
        self.epoch_attributes = epoch_attributes

    def before_train_epoch(self, runner):
        """Method called before each training epoch.

        It checks if the current epoch is in the `epoch_attributes` mapping and
        applies the corresponding attribute changes to the model.
        """
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        if epoch in self.epoch_attributes:
            for key, value in self.epoch_attributes[epoch].items():
                rsetattr(model.head, key, value)
                runner.logger.info(
                    f'Change model.head.{key} to {rgetattr(model.head, key)}')