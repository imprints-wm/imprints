import sys
import watermark_removal_works.SLBR.SLBR_Visible_Watermark_Removal.src.models as models


class slbr:
    def __init__(self, args, device) -> None:
        self.device = device
        self.argx = Argument(args.model_path)
        self.model = self.load_model()

    def load_model(self):
        Machine = models.__dict__[self.argx.models](args=self.argx)
        return Machine.model.eval().to(self.device)

    def resolve(self, outputs):
        return outputs[0][0], outputs[1][0]

    def endtoend_func(self, inputs, mask_eot):
        outputs = self.model(inputs)
        guess_images, guess_mask = outputs[0][0], outputs[1][0]
        expanded_guess_mask = mask_eot.repeat(1, 3, 1, 1)
        reconstructed_pixels = guess_images * expanded_guess_mask
        reconstructed_images = (
            inputs * (1 - expanded_guess_mask) + reconstructed_pixels
        )
        return reconstructed_images

class Argument(object):
    def __init__(
        self,
        root="../../watermark_removal_works/SLBR",
        test_dir="../../datasets/CLWD/test",
        models="slbr",
        crop_size=256,
        model_path="../../watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar"
    ):
        self.root = root
        self.test_dir = test_dir
        self.models = models
        self.crop_size = crop_size
        self.resume = model_path
        self.name = "slbr_v1"
        self.nets = "slbr"
        self.test_batch = 1
        self.preprocess = "resize"
        self.input_size = 256
        self.bg_mode = "res_mask"
        self.mask_mode = "cat"
        self.sim_metric = "cos"
        self.k_center = 2
        self.k_skip_stage = 3
        self.k_refine = 3
        self.project_mode = "simple"
        self.use_refine = True
        self.no_flip = True
        self.evaluate = True
        self.checkpoint = "checkpoint"
        self.lr = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0
        self.hl = False
        self.lambda_style = 0
        self.sltype = "vggx"
        self.lambda_content = 0
        self.lambda_iou = 0
        self.lambda_primary = 0.01
        self.start_epoch = 0
        self.schedule = [5, 10]
        self.gamma = 0.1
        self.gan_norm = False
        self.dataset_dir = "../../datasets/CLWD"
        self.lambda_mask = 1


def norm(x):
    return x


def denorm(x):
    return x
