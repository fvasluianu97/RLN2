# cleaned version of the testing script from Retinexformer (https://github.com/caiyuanhao1998/Retinexformer)
# without self-ensembling or lighting correction based on GT images
import argparse
import lpips
import numpy as np
import os
import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import yaml
from basicsr.models import create_model
from basicsr.utils.options import parse
from glob import glob
from skimage import img_as_ubyte
from tqdm import tqdm

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def pad_if_needed(img):
    h, w, _ = img.shape

    if h % 32 != 0 or w % 32 != 0:
        H = (h // 32 + 1) * 32
        W = (w // 32 + 1) * 32
        new_img = np.zeros([H, W, 3], np.uint8)
        new_img[:h, :w, :] = img
        print("PAD:", new_img.shape)
        return new_img
    else:
        return img




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image Enhancement using RLN2')
    parser.add_argument('--mode', type=str, default="model", help='GPU devices.')
    parser.add_argument('--result_dir', default='./RLN2_results/',
                        type=str, help='Directory for results')
    parser.add_argument(
        '--opt', type=str, default='Options/RLN2-Lf.yml', help='Path to option YAML file.')
    parser.add_argument('--weights', default='checkpoints/RLN2-Lf/best_psnr_20.51_5600.pth',
                        type=str, help='Path to weights')
    parser.add_argument('--dataset', default='CL3AN', type=str,
                        help='Test Dataset')
    parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
    parser.add_argument('--H', type=int, default=1080, help='Image Height.')
    parser.add_argument('--W', type=int, default=1440, help='Image Width.')

    args = parser.parse_args()

    gpu_list = ','.join(str(x) for x in args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    yaml_file = args.opt
    weights = args.weights
    print(f"dataset {args.dataset}")

    opt = parse(args.opt, is_train=False)
    opt['dist'] = False

    x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
    s = x['network_g'].pop('type')

    model_restoration = create_model(opt).net_g
    checkpoint = torch.load(weights)
    # model_restoration.load_state_dict(checkpoint['params'])

    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)


    print("===>Testing using weights: ", weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()


    # create dirs for results
    dataset = args.dataset
    config = os.path.basename(args.opt).split('.')[0]
    checkpoint_name = os.path.basename(args.weights).split('.')[0]

    result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
    os.makedirs(result_dir, exist_ok=True)

    # save results per image
    psnr = []
    ssim = []
    lpips_arr = []

    # load images
    gt_paths = [f for f in os.listdir("datasets/CL3AN/Testing/GT") if f.endswith('.png')]
    in_paths = [f for f in glob("datasets/CL3AN/Testing/**", recursive=True) if f.endswith("_IN.png")]

    path_dicts = {}
    for in_path in in_paths:
        img_id = in_path.split('/')[-1].split('_')[0]
        gt_path = "datasets/CL3AN/GT/{}_GT.png".format(img_id)
        path_dicts[in_path] = gt_path

    # set up LPIPS and save LPIPS per image
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn = loss_fn.cuda()
    lpips_values = {}
    
    
    with torch.inference_mode():
        for inp_path, tar_path in tqdm(path_dicts.items()) :#  tqdm(zip(input_paths, target_paths), total=len(target_paths)):

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = utils.load_img_resize(inp_path, (args.W, args.H))
            target = utils.load_img_resize(tar_path, (args.W, args.H))
            h, w, _ = target.shape

            # img = np.float32(pad_if_needed(img)) / 255.
            img = np.float32(img) / 255.
            target = np.float32(target) / 255.

            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()
            gt = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).cuda()

            if args.mode == "model":
                # Padding in case images are not multiples of 4
                factor = 32
                b, c, h, w = input_.shape
                H, W = ((h + factor) // factor) * \
                       factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

                restored = model_restoration(input_)
            elif args.mode == "inputs":
                restored = input_

            restored = restored[:, :, :h, :w]
            restored_sym = 2 * (restored - 0.5)
            gt_sym = 2 * (gt - 0.5)
            lpips_dist = loss_fn.forward(restored_sym, gt_sym).item()

            lpips_arr.append(lpips_dist)
            lpips_values[inp_path] = lpips_dist
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))

            # save image to results dir
            utils.save_img((os.path.join(result_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))


    psnr = np.mean(np.array(psnr))
    ssim = np.mean(np.array(ssim))
    lpips_avg = np.mean(np.array(lpips_arr))

    sorted_lpips_values = dict(sorted(lpips_values.items(), key=lambda item: item[1]))
    for k, v in sorted_lpips_values.items():
        print("%s: %f" % (k, v))
    
    print("PSNR: %f " % (psnr))
    print("SSIM: %f " % (ssim))
    print("LPIPS: %f " % (lpips_avg))