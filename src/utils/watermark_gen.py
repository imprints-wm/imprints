import cv2
import os
import random
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from .image_process import load_image

FONT_PATH = "/usr/share/fonts/truetype"


def gen_black_bg(path="./", name="black_bg.png"):
    a = np.full((3, 256, 256), 0.0)
    a = np.transpose(a, (1, 2, 0)) * 255
    cv2.imwrite(path + name, a)


def all_files(path=FONT_PATH, file_type=".ttf"):  # 生成path路径下全部file_type类型文件绝对路径列表
    f_list = []

    def files_list(father_path):
        sub_path = os.listdir(father_path)  # 读取父路径下全部文件或文件夹名称
        for sp in sub_path:
            full_sub_path = os.path.join(father_path, sp)  # 生成完整子路径
            if os.path.isfile(full_sub_path):  # 判断是否为文件
                file_name, post_name = os.path.splitext(full_sub_path)  # 获取文件后缀名
                if post_name == file_type:
                    f_list.append(file_name + post_name)
            else:  # 如果是文件夹，递归调用
                files_list(full_sub_path)

    files_list(path)
    return f_list  # 返回路径列表


def draw_text(text="usslab", size=25, save_path="./", save_name="txt_wm.png"):
    if not os.path.exists("./black_bg.png"):
        gen_black_bg()
    image = Image.open("./black_bg.png")
    draw = ImageDraw.Draw(image)
    font_lis = all_files()
    # font = ImageFont.truetype(font_lis[0], size)
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', size)


    # stroke_width: The width of the text stroke. Default: 0.
    # TODO: make stroke_width an input parameter.
    # length = draw.textlength(text, font)
    width, height = draw.textsize(text, font)
    print(width, height)
    draw.text(((256-width)//2, (256-height)//2), text, font=font, align="left", stroke_width=0, fill=(204, 204, 204))
    # draw.text(((255-length)//2, 90), text, font=font, align="left", stroke_width=1, fill=(128+64+16, 128, 128))

    
    # stroke_width = 4
    # char_spacing = 5
    # anchor = (5, 40)
    # for idx, c in enumerate(text):
    #     draw.text(anchor, c, font=font, align="left", stroke_width=stroke_width)
    #     anchor = (anchor[0]+draw.textlength(c, font)+char_spacing, anchor[1])
    
    image.save(save_path + save_name)


def gen_wm(text="S&P", text_size=42, standard_norm=False, device=torch.device("cuda:0")):
    image_size = 256
    # if not os.path.exists("./txt_wm.png"):
    draw_text(text=text, size=text_size)

    # should not only load a gray-scale image, but also make it binary. Here the threshold is 0.2
    mask = (load_image(path="./txt_wm.png", gray=True)[1]>0.3).type(torch.float32).to(device)
    
    # black_bg = load_image(path="./black_bg.png", standard_norm=standard_norm)[1].to(device)
    # patch = np.random.uniform(low=0.0, high=1.0, size=(3, image_size, image_size)).astype(np.float32)
    # patch = torch.from_numpy(patch).to(device)
    # patch = patch * (mask != 0) + black_bg
    # patch = load_image(path="./txt_wm.png", gray=False)[1].to(device)
    
    patch = cv2.imread("./txt_wm.png")
    patch = (torch.from_numpy(patch)/255.0).permute(2,0,1).to(device)
    print(patch.size())

    return patch[[2,1,0], :, :], mask

def load_logo(logo_path, standard_norm=False, device=torch.device("cuda:0")):
    image_size = 256
    scale_ratio = 0.5
    image = np.zeros([image_size, image_size, 4])

    # should not only load a gray-scale image, but also make it binary. Here the threshold is 0.2
    # mask = (load_image(path=logo_path, gray=True)[1]>0.3).type(torch.float32).to(device)
    
    patch = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    width = int(image_size*scale_ratio)
    patch_scaled = preprocess_logo(patch, width, width)

    padding = (image_size-width) // 2
    image[padding:padding+width, padding:padding+width, :] = patch_scaled
    # patch_scaled = patch_scaled + image[]

    # print(image.shape)
    patch_rgb = (torch.from_numpy(image[:,:,:3])/255.0).permute(2,0,1).type(torch.float32).to(device)
    
    if os.path.basename(logo_path) in ['6.png', '38.png']:
        mask = (torch.from_numpy((image[:,:,:3].sum(axis=2, keepdims=True)>20)*1.0)).permute(2,0,1).type(torch.float32).to(device)
    else:   
        mask = (torch.from_numpy(image[:,:,3:])/255.0).permute(2,0,1).type(torch.float32).to(device)
    
    # mask = (torch.from_numpy(image[:,:,3:])/255.0).permute(2,0,1).type(torch.float32).to(device)
    
    # print(mask.shape)
    patch_rgb = patch_rgb*mask
    return patch_rgb[[2,1,0], :, :], mask


def preprocess_logo(im, target_height, target_width):
    height, width = im.shape[:2]  # 取彩色图片的长、宽。

    ratio_h = height / target_height
    ration_w = width / target_width

    ratio = max(ratio_h, ration_w)

    # 缩小图像  resize(...,size)--size(width，height)
    size = (int(width / ratio), int(height / ratio))
    shrink = cv2.resize(im, size, interpolation=cv2.INTER_AREA)  # 双线性插值
    BLACK = [0, 0, 0]  # 修改该值可以将放大部分填成任意颜色

    a = (target_width - int(width / ratio)) / 2
    b = (target_height - int(height / ratio)) / 2

    constant = cv2.copyMakeBorder(shrink, int(b), int(b), int(a), int(a), cv2.BORDER_CONSTANT, value=BLACK)
    constant = cv2.resize(constant, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return constant

# a = np.random.randint(low=0,high=2,size=(1,4,4))
# b = np.random.randint(low=0,high=2,size=(3,4,4))
# a,b=torch.from_numpy(a),torch.from_numpy(b)
# print(a)
# print(b)
# print(a*b)