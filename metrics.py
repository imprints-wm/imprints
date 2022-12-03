import pytorch_ssim
from PIL import Image
import torch
import math
from scipy.ndimage import gaussian_filter
import numpy
from numpy.lib.stride_tricks import as_strided as ast
import cv2
import os.path as osp
import os
from tqdm import tqdm
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] =  "0"

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
"""
Hat tip: http://stackoverflow.com/a/5078155/1828289
"""

def mse(img1, img2):
    mse=numpy.mean( (img1 - img2) ** 2 )
    return mse

# def block_view(A, block=(3, 3)):
#     """Provide a 2D block view to 2D array. No error checking made.
#     Therefore meaningful (as implemented) only for blocks strictly
#     compatible with the shape of A."""
#     # simple shape and strides computations may seem at first strange
#     # unless one is able to recognize the 'tuple additions' involved ;-)
#     shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
#     strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
#     return ast(A, shape= shape, strides= strides)
# def ssim(img1, img2, C1=0.01**2, C2=0.03**2):

#     bimg1 = block_view(img1, (4,4))
#     bimg2 = block_view(img2, (4,4))
#     s1  = numpy.sum(bimg1, (-1, -2))
#     s2  = numpy.sum(bimg2, (-1, -2))
#     ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
#     s12 = numpy.sum(bimg1*bimg2, (-1, -2))

#     vari = ss - s1*s1 - s2*s2
#     covar = s12 - s1*s2

#     ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
#     return numpy.mean(ssim_map)

def four_metrics(img_A, img_B, mask):
    '''
    ::params img_A  [0, 255]
    ::params img_B  [0, 255]
    ::params mask   0 or 1
    '''
    img_A_ts = (torch.from_numpy(img_A).float().unsqueeze(0)/255.0).cuda()
    img_B_ts = (torch.from_numpy(img_B).float().unsqueeze(0)/255.0).cuda()

    psnr_x_xr = psnr(img_A, img_B)
    ssim_x_xr = pytorch_ssim.ssim(img_A_ts, img_B_ts) 

    mse_x_xr = mse(img_A, img_B)
    rmse_x_xr = numpy.sqrt(mse_x_xr) 
    mse_in = mse(img_A * mask, img_B * mask) \
                * mask.shape[0] * mask.shape[1] * mask.shape[2] \
                /(numpy.sum(mask) + 1e-6)
    rmse_in = numpy.sqrt(mse_in)

    return psnr_x_xr, ssim_x_xr, rmse_x_xr, rmse_in

def eval_single_img(
        watermark_free_img_path: str, 
        watermarked_img_path: str, 
        result_img_path: str, 
        mask_path: str):
    '''
    ::params watermark_free_img     path of the watermark-free image
    ::params watermarked_img        path of the watermarked image
    ::params result_img             path of the resultant image
    ::params mask_img               path of the mask
    '''
    mask = Image.open(mask_path)
    mask = numpy.asarray(mask)/255.0

    watermark_free_img = cv2.imread(watermark_free_img_path)
    watermarked_img = cv2.imread(watermarked_img_path)
    result_img = cv2.imread(result_img_path)

    # perfect removal
    psnr_xw_x, ssim_xw_x, rmse_xw_x, rmse_w_xw_x = four_metrics(watermarked_img,
                                                                watermark_free_img,
                                                                mask)

    # restoration performance
    psnr_x_xr, ssim_x_xr, rmse_x_xr, rmse_w_x_xr = four_metrics(watermark_free_img,
                                                                result_img,
                                                                mask)

    # removal performance
    psnr_xw_xr, ssim_xw_xr, rmse_xw_xr, rmse_w_xw_xr = four_metrics(watermarked_img,
                                                                result_img,
                                                                mask)

    return [[psnr_xw_x, ssim_xw_x.item(), rmse_xw_x, rmse_w_xw_x],
            [psnr_x_xr, ssim_x_xr.item(), rmse_x_xr, rmse_w_x_xr],
            [psnr_xw_xr, ssim_xw_xr.item(), rmse_xw_xr, rmse_w_xw_xr]]

# # Demo
# idx = 7
# results = eval_single_img(
#                 watermark_free_img_path = f'./out_test_pics/slbr/Watermark_free_image/{idx}.png', 
#                 watermarked_img_path = f'./adv_wm_pics/slbr/Watermarked_image/{idx}.png', 
#                 result_img_path = f'./out_test_pics/slbr/Reconstruct_image/{idx}.png', 
#                 mask_path = f'./out_test_pics/slbr/Mask/{idx}.png'    
#             )

# print(results[0])
# print(results[1])


# input a directory, average the metrics, give a summary
def eval_directory(directory: str):
    '''
    ::params directory  a directory containing subdirs of
                                            1. Watermark_free_image/, 
                                            2. Watermarked_image/, 
                                            3. Reconstruct_image/, 
                                            4. Mask/
    '''
    xw_x = []
    x_xr = []
    xw_xr = []

    files = os.listdir(osp.join(directory, 'Watermark_free_image'))
    for i in tqdm(files):
        results = eval_single_img(
                watermark_free_img_path = osp.join(directory, f'Watermark_free_image/{i}'), 
                watermarked_img_path = osp.join(directory, f'Watermarked_image/{i}'), 
                result_img_path = osp.join(directory, f'Reconstruct_image/{i}'), 
                mask_path = osp.join(directory, f'Mask/{i}')  
            )
        xw_x.append(results[0])
        x_xr.append(results[1])
        xw_xr.append(results[2])
        
    xw_x = numpy.vstack(xw_x)
    xw_x_mean = xw_x.mean(axis = 0)
    xw_x_std = xw_x.std(axis = 0, ddof=1)

    x_xr = numpy.vstack(x_xr)
    x_xr_mean = x_xr.mean(axis = 0)
    x_xr_std = x_xr.std(axis = 0, ddof=1)

    xw_xr = numpy.vstack(xw_xr)
    xw_xr_mean = xw_xr.mean(axis = 0)
    xw_xr_std = xw_xr.std(axis = 0, ddof=1)

    print('='*30+' summary '+'='*30)
    print('Total imgs: ', len(files), '\n')
    print('Perfect removal: wm imgs v.s. wm-free imgs')
    print('mean PSNR: {:.2f}\tSSIM: {:.4f}\tRMSE: {:.4f}\tRMSE_w: {:.4f}'\
            .format(*list(xw_x_mean)))
    print('sstd PSNR: {:.2f}\t\tSSIM: {:.4f}\tRMSE: {:.4f}\tRMSE_w: {:.4f}'\
            .format(*list(xw_x_std)))
    
    print()    
    print('Restoration: wm-free imgs v.s. result imgs')
    print('mean PSNR↓: {:.2f}\tSSIM↓: {:.4f}\tRMSE↑: {:.4f}\tRMSE_w↑: {:.4f}'\
            .format(*list(x_xr_mean)))
    print('sstd PSNR : {:.2f}\tSSIM : {:.4f}\tRMSE : {:.4f}\tRMSE_w : {:.4f}'\
            .format(*list(x_xr_std)))

    print()
    print('Removal: wm imgs v.s. result imgs')
    print('mean PSNR↑: {:.2f}\tSSIM↑: {:.4f}\tRMSE↓: {:.4f}\tRMSE_w↓: {:.4f}'\
            .format(*list(xw_xr_mean)))
    print('sstd PSNR : {:.2f}\tSSIM : {:.4f}\tRMSE : {:.4f}\tRMSE_w : {:.4f}'\
            .format(*list(xw_xr_std)))
    print('='*30+' summary '+'='*30)

# # Demo
# eval_directory('/home/imprints/Imprints/out_test_pics/slbr')

def eval_single_pair(
        path_img_A: str, 
        path_img_B: str, 
        path_mask: str):
    '''
    ::params img_A      path of the first image
    ::params img_B      path of the second image
    ::params mask       path of the mask
    '''
    mask = Image.open(path_mask)
    mask = numpy.asarray(mask)/255.0

    img_A = cv2.imread(path_img_A)
    img_B = cv2.imread(path_img_B)

    _psnr, _ssim, _rmse, _rmse_w = four_metrics(img_A,
                                                img_B,
                                                mask)

    return [_psnr, _ssim.item(), _rmse, _rmse_w]


# compare two directories of images
def eval_compare(dir_img_A: str, dir_img_B: str, dir_mask: str):
    '''
    ::params dir_img_A  a directory of images
    ::params dir_img_B  a directory of images
    ::params dir_mask   a directory of corresponding masks

    Usage:  eval_compare('Watermarked_image/', 'Reconstruct_image/', 'Mask/')
    '''
    all_results = []
    files = os.listdir(dir_img_B)
    for i in tqdm(files):
        results = eval_single_pair(
                path_img_A = osp.join(dir_img_A, i),
                path_img_B = osp.join(dir_img_B, i).replace("jpg", "png"),
                path_mask = osp.join(dir_mask, i).replace("jpg", "png")
            )
        all_results.append(results)
        
    all_results = numpy.vstack(all_results)
    _mean = all_results.mean(axis = 0)
    _std = all_results.std(axis = 0, ddof=1)

    print('='*30+' summary '+'='*30)
    print('Comparing {} & {}'.format(dir_img_A, dir_img_B))
    print('Total imgs: ', len(files), '\n')

    print('mean PSNR↑: {:.2f}\tSSIM↑: {:.4f}\tRMSE↓: {:.4f}\tRMSE_w↓: {:.4f}'\
            .format(*list(_mean)))
    print('sstd PSNR : {:.2f}\tSSIM : {:.4f}\tRMSE : {:.4f}\tRMSE_w : {:.4f}'\
            .format(*list(_std)))
    print('='*30+' summary '+'='*30)


if __name__ == '__main__':
    # TODO: parse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--eval_path', type=str, help='path to watermarked images')
    argparser.add_argument('--stage', type=str, help='which dir to be evaluated', default='1,2,3,4,5')

    #argparser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    args = argparser.parse_args()
    eval_path = args.eval_path

    if '1' in list(args.stage):
        if os.path.exists(f"{eval_path}/out_test_pics/slbr/"):
            print("*-@"*10 + "Reconstruct vs. watermarked" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermarked_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/slbr/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

            print("*-@"*10 + "Reconstruct vs. watermark-free" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermark_free_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/slbr/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

    # =================================================================
    if '2' in list(args.stage):
        if os.path.exists(f"{eval_path}/out_test_pics/bvmr/"):
            print("*-@"*10 + "Reconstruct vs. watermarked" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermarked_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/bvmr/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

            print("*-@"*10 + "Reconstruct vs. watermark-free" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermark_free_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/bvmr/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

    # =================================================================
    if '3' in list(args.stage):
        if os.path.exists(f"{eval_path}/out_test_pics/split/"):
            print("*-@"*10 + "Reconstruct vs. watermarked" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermarked_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/split/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

            print("*-@"*10 + "Reconstruct vs. watermark-free" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermark_free_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/split/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

    # =================================================================
    if '4' in list(args.stage):
        if os.path.exists(f"{eval_path}/out_test_pics/slbr_with_real_mask/"):
            print("*-@"*10 + "non-blind Reconstruct vs. watermarked" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermarked_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/slbr_with_real_mask/slbr/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

            print("*-@"*10 + "non-blind Reconstruct vs. watermark-free" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermark_free_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/slbr_with_real_mask/slbr/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

    # =================================================================
    if '5' in list(args.stage):
        if os.path.exists(f"{eval_path}/out_test_pics/multi/"):
            print("*-@"*10 + "non-blind Reconstruct vs. watermarked" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermarked_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/multi/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

            print("*-@"*10 + "non-blind Reconstruct vs. watermark-free" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermark_free_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/multi/Reconstruct_image/', 
                        dir_mask = f'{eval_path}/Mask/')

    # =================================================================
    if '6' in list(args.stage):
        if os.path.exists(f"{eval_path}/out_test_pics/commer/"):
            print("*-@"*10 + "non-blind Reconstruct vs. watermarked" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermarked_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/commer/', 
                        dir_mask = f'{eval_path}/Mask/')

            print("*-@"*10 + "non-blind Reconstruct vs. watermark-free" + "*-@"*10)
            eval_compare(dir_img_A = f'{eval_path}/Watermark_free_image/',
                        dir_img_B = f'{eval_path}/out_test_pics/commer/', 
                        dir_mask = f'{eval_path}/Mask/')