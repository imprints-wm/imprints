import cv2
import glob
import numpy as np
import torch


def frame_the_char(char_ts):
    non0 = torch.nonzero(char_ts)
    left = torch.min(non0[:, 2]).item()
    right = torch.max(non0[:, 2]).item()
    top = torch.min(non0[:, 1]).item()
    down = torch.max(non0[:, 1]).item()

    return left, down, right, top


strings = [
    "IEEE", "S&P"
]

save_path = "./exp_data/strings/logo_log2_2_0.7/"

for sidx, target_string in enumerate(strings):
    padding = 8
    lr_padding = 10
    root_dir = './ckpt_wm/slbr/log/charset_1129/'

    all_char = []
    all_length = lr_padding
    start_pos = [lr_padding]

    for char in target_string[:]:
        if char == ' ':
            all_length += (10+padding)
            start_pos.append(all_length)
            all_char.append(torch.zeros(3, 256, 10))
            continue

        if char == '/':
            char = 'slash'

        cur_char_path = glob.glob(root_dir + '{}_*/wm_adv_0.pt'.format(char))
        try:
            cur_char = torch.load(cur_char_path[0])
        except Exception as e:
            print("ERROR:", char)
            # exit()
        cur_mask = cur_char.sum(dim=0, keepdim=True) > 0
        left, down, right, top = frame_the_char(cur_mask)
        all_length += (right-left+padding)
        start_pos.append(all_length)

        if char in list('ypg'):
            downshift = 6
            sheet = torch.zeros_like(cur_char)
            sheet[:, downshift:, :] = cur_char[:, :-downshift, :]
            cur_char = sheet
            print("shifted")

        all_char.append(cur_char[:, :, left:right])

        print(char, left, down, right, top)

    all_length_pad = all_length + lr_padding
    print(all_length)
    print(all_length_pad)
    print(start_pos)

    whole_string = torch.zeros([3, 256, all_length_pad])
    for idx, char in enumerate(all_char):
        whole_string[:, :, start_pos[idx]:start_pos[idx]+char.size(-1)] = char

    print(whole_string.size())

    whole_string = whole_string.cpu().numpy()
    whole_string = np.transpose(whole_string, (1, 2, 0)) * 255
    alpha = (whole_string.sum(axis=-1, keepdims=True) > 0)*int(255*1)
    print(whole_string.shape, alpha.shape)
    whole_string = np.concatenate([whole_string, alpha], axis=-1)
    cv2.imwrite(save_path+f'{sidx+1}.png', whole_string)
