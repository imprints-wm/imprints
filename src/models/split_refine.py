import torch
import watermark_removal_works.split_then_refine.deep_blind_watermark_removal.scripts.models as models
import watermark_removal_works.split_then_refine.deep_blind_watermark_removal.scripts.machines as machines

class split:
    def __init__(self,args,device) -> None:
        self.device=device
        self.model=self.load_model(args)
    def load_model(self,args):
        model_ = models.__dict__["vvv4n"]().to(self.device)
        model_.load_state_dict(torch.load(args.model_path)["state_dict"])

        return model_.eval()

    def resolve(self, outputs):
        return outputs[0][0], outputs[1]

    def endtoend_func(self, inputs):
        outputs = self.model(inputs)
        guess_images, guess_mask = outputs[0][0], outputs[1]
        expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
        reconstructed_pixels = guess_images * expanded_guess_mask
        reconstructed_images = (
            inputs * (1 - expanded_guess_mask) + reconstructed_pixels
        )
        return reconstructed_images