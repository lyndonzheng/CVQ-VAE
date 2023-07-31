import lpips
import numpy as np
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch

from image_folder import make_dataset
from fid_score import calculate_fid_given_paths

parser = argparse.ArgumentParser(description='Image quality evaluations on the dataset')
parser.add_argument('--gt_path', type=str, default='../results/', help='path to original gt data')
parser.add_argument('--g_path', type=str, default='../results.', help='path to the generated data')
parser.add_argument('--num_test', type=int, default=0, help='how many examples to load for testing')

args = parser.parse_args()
lpips_vgg = lpips.LPIPS(net='vgg')

def calculate_score(img_gt, img_test):
    """
    function to calculate the image quality score
    :param img_gt: original image
    :param img_test: generated image
    :return: mae, ssim, psnr
    """

    l1loss = np.mean(np.abs(img_gt-img_test))

    psnr_score = psnr(img_gt, img_test, data_range=1)

    ssim_score = ssim(img_gt, img_test, multichannel=True, data_range=1, win_size=11)

    lpips_dis = lpips_vgg(torch.from_numpy(img_gt).permute(2, 0, 1), torch.from_numpy(img_test).permute(2, 0, 1), normalize=True)

    return l1loss, ssim_score, psnr_score, lpips_dis.data.numpy().item()


if __name__ == '__main__':
    gt_paths, gt_size = make_dataset(args.gt_path)
    g_paths, g_size = make_dataset(args.g_path)
    fid_score = calculate_fid_given_paths([args.gt_path, args.g_path], batch_size=50, cuda=True, dims=2048)

    l1losses = []
    ssims = []
    psnrs = []
    lpipses = []

    size = args.num_test if args.num_test > 0 else gt_size

    for i in range(size):
        gt_img = Image.open(gt_paths[i]).convert('RGB')
        gt_numpy = np.array(gt_img).astype(np.float32) / 255.0
        
        g_img = Image.open(g_paths[i]).convert('RGB')
        g_numpy = np.array(g_img).astype(np.float32) / 255.0
        l1loss, ssim_score, psnr_score, lpips_score = calculate_score(gt_numpy, g_numpy)
        print(l1loss, ssim_score, psnr_score, lpips_score)
        l1losses.append(l1loss)
        ssims.append(ssim_score)
        psnrs.append(psnr_score)
        lpipses.append(lpips_score)

    print('{:>10},{:>10},{:>10},{:>10},{:>10}'.format('l1loss', 'SSIM', 'PSNR', 'LPIPS', 'FID-Score'))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.mean(l1losses), np.mean(ssims), np.mean(psnrs), np.mean(lpipses), np.mean(fid_score)))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.var(l1losses), np.var(ssims), np.var(psnrs), np.var(lpipses)))