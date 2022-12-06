This is the repository of IMPRINTS, the idelible watermarking framework.

# Preparation

## Training and testing data

First, Download the colored large-scale watermark dataset (CLWD) to `datasets/` from the link below:

Download the CLWD dataset: [https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij/view?usp=sharing](https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij/view?usp=sharing)

Second, uncompress the `CLWD.rar` file `datasets/`, making `datasets/CLWD/`

## Watermark removal models

Due to space limitation, we cannot upload the pretrain model for *split-and-refine*. You can download the pretrain model from the link below and place the `27kpng_model_best.pth.tar` file in `watermark_removal_works/split_then_refine`.

The official repository of *split-and-refine*: [https://github.com/vinthony/deep-blind-watermark-removal](https://github.com/vinthony/deep-blind-watermark-removal)


# Watermark optimization (training)

IMPEINTS supports two types of watermarks, logo and strings. For logo, it means you have to provide a `png` image that contains an *alpha* channel denoting the mask of the watermark. For string, it means you just have to provide a digital string input, IMPRINTS will convert the input into an image for you.

## logo training

The main code for logo training is `train_logo.py`. Below is an example for training a logo in the CLWD-*test*, by specifying the path to the logo with `--logo_path`.

```shell
    python train_logo.py \
    --model slbr \
    --model_path ./watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar \
    --output_dir ./adv_wm_pics/ \
    --logo_path ./datasets/CLWD/watermark_logo/test_color/1.png \
    --epoch 50 \
    --standard_transform 1 \
    --eps 0.2 \
    --log ./ckpt_wm/slbr/log/ \
    --gpu 0,1,2,3 \
    --batch_size 20
```

## string training

The main code for string training is `train_char.py`. Below is an example for training a character or string input. The only difference is that the `--logo_path` is replaced by `--text` and `--text_size`.

```shell
    python train.py \
    --model slbr \
    --model_path ./watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar \
    --output_dir ./adv_wm_pics/ \
    --text "IMPRINTS" \
    --text_size 50 \
    --epoch 50 \
    --standard_transform 1 \
    --eps 0.2 \
    --log ./ckpt_wm/slbr/log/ \
    --gpu 0,1,2,3 \
    --batch_size 20
```

# Watermark deployment

Suppose you have trained a directory of logos or strings. Two formats are supported:

1. a directory of `.png` images, each one is a watermark, naive or IMPRITNS'
2. a directory of directories generated by IMPRINTS, i.e.,

```shell
└── 2022-12-XX-HH-MM-SS
    ├── wm_adv_0.pt
    ├── wm_adv_best.png
    ├── wm_adv_latest.png
    └── wm_adv_mask_end.pt
```

Then run `deploy_many_logos.sh` can deploy the watermarks on the host images of CLWD-*test*. P.S. specify the saving path first.

```shell
bash deploy_many_logos.sh \
    ${save_dir} \
    ./ckpt_wm/slbr/log/ \
    False \
    False \
    False \
    False \
    False \
    0.6 \
    1 \
```

See `deploy_many_logos.sh` and `deploy_many_logos.py` for detailed definitions of the parameters.

# Char concatenation

You can use `concat.py` to concatenate you pre-trained characters into a `.png` image, such that you can use it as a logo for deployment afterwards.
