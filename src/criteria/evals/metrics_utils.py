import torch
from torchmetrics.utilities.distributed import gather_all_tensors
from typing import Any, List, Optional

def all_gather_on_cuda(tensor: torch.Tensor, *args: Optional[Any], **kwargs: Optional[Any]) -> List[torch.Tensor]:
    original_device = tensor.device
    return [
        _tensor.to(original_device)
        for _tensor in gather_all_tensors(tensor.cuda(non_blocking=True), *args, **kwargs)
    ]