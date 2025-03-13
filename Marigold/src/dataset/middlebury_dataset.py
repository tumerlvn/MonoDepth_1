import torch
from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode

class MiddleburyDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            min_depth=1e-3,  # Adjust based on Middlebury depth range
            max_depth=40.0,  # Adjust based on Middlebury depth range
            has_filled_depth=False,
            name_mode=DepthFileNameMode.rgb_id,  # Adjust based on Middlebury file naming
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode Middlebury depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = super()._get_valid_mask(depth)
        
        return valid_mask