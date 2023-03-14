import argparse

class Options:
    """docstring for Options"""

    def __init__(self, is_train=True, deploy=False):
        if deploy:
            self.parser = argparse.ArgumentParser(
                description="Set some necessary arguments for generating watermarked images"
            )
            self.deploy()
        else:
            if is_train:
                self.parser = argparse.ArgumentParser(
                    description="Set some necessary arguments before training"
                )
                self.train()
            else:
                self.parser = argparse.ArgumentParser(
                    description="Set some necessary arguments before testing"
                )
                self.test()

    def train(self):

        # Target model(s) information
        self.parser.add_argument(
            "--model",
            type=str,
            metavar="MODEL",
            default="bvmr",
            help="training based model (bvmr, slbr, split)",
        )
        self.parser.add_argument(
            "--model_path", 
            type=str, 
            metavar="PATH OF MODEL", 
            help="Path to the pretrained model"
        )

        # hyperparameters during training
        self.parser.add_argument(
            "--epoch",
            type=int,
            metavar="Training Epoch",
            default=1,
            help="Training epoch nums",
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            metavar="Alpha",
            default=4 / 255,
            help="Training learning rate",
        )
        self.parser.add_argument(
            "--eps",
            type=float,
            metavar="Epsilon of noise",
            default=0.01,
            help="Epsilon of Watermark noise",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            metavar="Training Batch Size",
            default=12,
            help="Batch Size of training data",
        )
        self.parser.add_argument(
            "--sample_num",
            type=int,
            metavar="Total Numbers of Training images",
            default=200,
            help="Total Numbers of Training images",
        )
        self.parser.add_argument(
            "--transparences",
            nargs="+",
            type=float,
            metavar="Transparences to be trained",
            help="Transparences list",
        )

        # Watermark arguments
        self.parser.add_argument(
            "--text",
            type=str,
            metavar="WM Text",
            default="S&P",
            help="Text of Watermark, short text may be better",
        )
        self.parser.add_argument(
            "--text_size",
            type=int,
            metavar="TEXT SIZE",
            default=40,
            help="Size of Watermark",
        )

        self.parser.add_argument(
            "--logo_path", 
            type=str, 
            metavar="PATH OF THE LOGO", 
            help="Path to the logo image"
        )        

        # Output arguments
        self.parser.add_argument(
            "--output_dir",
            type=str,
            metavar="Output Directory",
            default="./output_adv_images",
            help="Save Path of generated images, watermark and relevent masks",
        )

        # others
        self.parser.add_argument(
            "--gpu",
            type=str,
            metavar="GPU",
            default="0,1,2,3",
            help="GPU list",
        )
        self.parser.add_argument(
            "--standard_transform",
            type=int,
            metavar="TransForm",
            default=0,
            help="if pictures need to be normalized to [0,1] then true else false",
        )
        self.parser.add_argument(
            "--log", 
            type=str, 
            metavar="PATH OF log and resultant images", 
            help="PATH OF log and resultant images"
        )
        # =============================
        self.parser.add_argument(
            "--change_pos",
            type=str,
            metavar="change position",
            default="false",
            help="change position",
        )     
        self.parser.add_argument(
            "--change_siz",
            type=str,
            metavar="change size",
            default="false",
            help="change size",
        )  
        self.parser.add_argument(
            "--change_ang",
            type=str,
            metavar="change angle",
            default="false",
            help="change angle",
        )  
        self.parser.add_argument(
            "--change_opa",
            type=str,
            metavar="change opacity",
            default="false",
            help="change opacity",
        )
        self.parser.add_argument(
            "--add_noise",
            type=str,
            metavar="change opacity",
            default="false",
            help="change opacity",
        ) 
        self.parser.add_argument(
            "--random_opa",
            type=str,
            metavar="randomized opacity",
            default="false",
            help="randomized opacity",
        ) 
        self.parser.add_argument(
            "--opa_value",
            type=float,
            metavar="change opacity",
            default=0.6,
            help="change opacity",
        )  
        self.parser.add_argument(
            "--size_value",
            type=float,
            metavar="change size",
            default=1,
            help="change size",
        )  


    def test(self):
        # Target model(s) information
        self.parser.add_argument(
            "--model",
            type=str,
            metavar="MODEL",
            default="bvmr",
            help="training based model (bvmr, slbr, split)",
        )
        self.parser.add_argument(
            "--model_path", 
            type=str, 
            metavar="PATH OF MODEL", 
            help="Path to the pretrained model"
        )

        # Input arguments
        self.parser.add_argument(
            "--input_dir",
            type=str,
            metavar="Input Directory",
            help="Save Path of refined images, predicted watermark and relevent masks",
        )
        self.parser.add_argument(
            "--bgs_dir",
            type=str,
            metavar="pure pictures Directory",
            help="for saving use",
        )

        # Output arguments
        self.parser.add_argument(
            "--output_dir",
            type=str,
            metavar="Output Directory",
            default="./output_adv_images",
            help="Save Path of refined images, predicted watermark and relevent masks",
        )

        self.parser.add_argument(
            "--gpu",
            type=str,
            metavar="GPU",
            default="0,1,2,3",
            help="GPU list",
        )
        self.parser.add_argument(
            "--standard_transform",
            type=int,
            metavar="TransForm",
            default=0,
            help="if pictures need to be normalized to [0,1] then true else false",
        )
        self.parser.add_argument(
            "--iteration",
            type=int,
            metavar="progress n iterations",
            default=1,
            help="how many iterations to process the images",
        )      

        pass
    
    def deploy(self):
        self.parser.add_argument(
            "--bgs_dir",
            type=str,
            metavar="pure pictures Directory",
            help="for saving use",
        )
        self.parser.add_argument(
            "--wm_path",
            type=str,
            metavar="path of watermark",
            help="for saving use",
        )
        self.parser.add_argument(
            "--mask_path",
            type=str,
            metavar="path of mask",
            help="for saving use",
        )
        self.parser.add_argument(
            "--output_dir",
            type=str,
            metavar="path of watermark",
            help="for saving use",
        )
        self.parser.add_argument(
            "--standard_transform",
            type=int,
            metavar="TransForm",
            default=0,
            help="if pictures need to be normalized to [0,1] then true else false",
        )
        pass