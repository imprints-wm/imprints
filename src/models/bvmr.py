import os
import sys
import torch

from watermark_removal_works.BVMR.visual_motif_removal.networks.baselines import UnetBaselineD
from watermark_removal_works.BVMR.visual_motif_removal.loaders.motif_dataset import MotifDS
from watermark_removal_works.BVMR.visual_motif_removal.utils.visualize_utils import run_net
from watermark_removal_works.BVMR.visual_motif_removal.utils.train_utils import load_globals, init_folders

# consts
ROOT_PATH = "../../watermark_removal_works/BVMR/demo_coco/checkpoints/demo_coco"
NET_PATH = "%s/net_baseline_200.pth" % ("../../watermark_removal_works/BVMR/demo_coco/checkpoints/demo_coco")

class bvmr:
    def __init__(self,args,device) -> None:
        self.device = device
        self.model=self.load_model(args.model_path)
    def load_model(self,model_path):
        opt = load_globals(ROOT_PATH, {}, override=False)
        net_baseline = UnetBaselineD(
        shared_depth=opt.shared_depth,
        use_vm_decoder=opt.use_vm_decoder,
        blocks=opt.num_blocks,
        )
        net_baseline.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        net_baseline = net_baseline.to(self.device)

        return net_baseline
    def resolve(self, outputs):
        return outputs[0], outputs[1]

    def endtoend_func(self, inputs):
        outputs = self.model(inputs)
        guess_images, guess_mask = outputs[0], outputs[1]
        expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
        reconstructed_pixels = guess_images * expanded_guess_mask
        reconstructed_images = (
            inputs * (1 - expanded_guess_mask) + reconstructed_pixels
        )
        return reconstructed_images