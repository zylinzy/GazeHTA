# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from typing import Dict, Optional, Union, Tuple

from mmengine.registry import MODELS
from torch import nn
import torch

TORCH_VERSION = torch.__version__

MODELS.register_module('Conv1d', module=nn.Conv1d)
MODELS.register_module('Conv2d', module=nn.Conv2d)
MODELS.register_module('Conv3d', module=nn.Conv3d)
MODELS.register_module('Conv', module=nn.Conv2d)

def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `conv_layer` cannot be found
    # in the registry, fallback to search `conv_layer` in the
    # mmengine.MODELS.
    with MODELS.switch_scope_and_registry(None) as registry:
        conv_layer = registry.get(layer_type)
    if conv_layer is None:
        raise KeyError(f'Cannot find {conv_layer} in registry under scope '
                       f'name {registry.scope}')
    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer



def get_norm() -> tuple:
    """A wrapper to obtain base classes of normalization layers from PyTorch or
    Parrots."""
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
    else:
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_


_BatchNorm, _InstanceNorm, SyncBatchNorm_ = get_norm()

class SyncBatchNorm(SyncBatchNorm_):  # type: ignore

    def _check_input_dim(self, input):
        if TORCH_VERSION == 'parrots':
            if input.dim() < 2:
                raise ValueError(
                    f'expected at least 2D input (got {input.dim()}D input)')
        else:
            super()._check_input_dim(input)

MODELS.register_module('BN', module=nn.BatchNorm2d)
MODELS.register_module('BN1d', module=nn.BatchNorm1d)
MODELS.register_module('BN2d', module=nn.BatchNorm2d)
MODELS.register_module('BN3d', module=nn.BatchNorm3d)
MODELS.register_module('SyncBN', module=SyncBatchNorm)
MODELS.register_module('GN', module=nn.GroupNorm)
MODELS.register_module('LN', module=nn.LayerNorm)
MODELS.register_module('IN', module=nn.InstanceNorm2d)
MODELS.register_module('IN1d', module=nn.InstanceNorm1d)
MODELS.register_module('IN2d', module=nn.InstanceNorm2d)
MODELS.register_module('IN3d', module=nn.InstanceNorm3d)


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """

    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    else:
        # Switch registry to the target scope. If `norm_layer` cannot be found
        # in the registry, fallback to search `norm_layer` in the
        # mmengine.MODELS.
        with MODELS.switch_scope_and_registry(None) as registry:
            norm_layer = registry.get(layer_type)
        if norm_layer is None:
            raise KeyError(f'Cannot find {norm_layer} in registry under '
                           f'scope name {registry.scope}')
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

MODELS.register_module('nearest', module=nn.Upsample)
MODELS.register_module('bilinear', module=nn.Upsample)
MODELS.register_module('deconv', module=nn.ConvTranspose2d)

def build_upsample_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
              deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        upsample = layer_type
    # Switch registry to the target scope. If `upsample` cannot be found
    # in the registry, fallback to search `upsample` in the
    # mmengine.MODELS.
    else:
        with MODELS.switch_scope_and_registry(None) as registry:
            upsample = registry.get(layer_type)
        if upsample is None:
            raise KeyError(f'Cannot find {upsample} in registry under scope '
                           f'name {registry.scope}')
        if upsample is nn.Upsample:
            cfg_['mode'] = layer_type
    layer = upsample(*args, **kwargs, **cfg_)
    return layer
