import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import img2tensor, padding
import numpy as np
import cv2


class Dataset_PairedScene(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedScene, self).__init__()
        self.opt = opt
        # file client (io backend) 文件客户端

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.basedir = opt['baseroot']
        self.square = opt['square'] if 'square' in opt else False

        self.read_dir = os.path.join(opt['baseroot'], opt['lphase'])

        self.gts = [f for f in os.listdir(os.path.join(opt['baseroot'], opt['lphase'], "GT")) if f.endswith('.png')]
        self.incr = [f for f in os.listdir(os.path.join(opt['baseroot'], opt['lphase'], "IN_CR")) if os.path.isdir(os.path.join(opt['baseroot'], opt['lphase'], "IN_CR", f))]
        self.insh = [f for f in os.listdir(os.path.join(opt['baseroot'], opt['lphase'], "IN_SH")) if os.path.isdir(os.path.join(opt['baseroot'], opt['lphase'], "IN_SH", f))]

        assert len(self.gts) == len(self.incr) and len(self.insh) == len(self.gts)

        if opt['phase'] == 'train':
            self.img_index = {}

            for gt in self.gts:
                img_k = gt.split("_")[0]

                gt_path = os.path.join(opt['baseroot'], opt['lphase'], "GT", gt)
                insh_path = os.path.join(opt['baseroot'], opt['lphase'], "IN_SH", img_k, "{}_1_IN.png".format(img_k))
                incr_paths = [os.path.join(opt['baseroot'], opt['lphase'], "IN_CR", img_k, f) for f in os.listdir(os.path.join(opt['baseroot'], opt['lphase'], "IN_CR", img_k)) if f.endswith('.png')]

                self.img_index[img_k] = {'gt': gt_path, 'insh': insh_path, 'incr': incr_paths}


        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

        if opt['maxsize'] is not None:
            self.max_size = opt['maxsize']
            print("Creating {} dataset with maxsize {}".format(opt['phase'], opt['maxsize']))
        else:
            self.max_size = None
            print("Creating {} dataset with original size".format(opt['phase']))

        if self.opt['phase'] != 'train':
            self.val_inputs = []
            for gt in self.gts:
                img_k = gt.split("_")[0]
                gt_path = os.path.join(opt['baseroot'], opt['lphase'], "GT", gt)
                insh_path = os.path.join(opt['baseroot'], opt['lphase'], "IN_SH", img_k,
                                         "{}_1_SH_IN.png".format(img_k))
                self.val_inputs.append((gt_path, insh_path))
                for f in os.listdir(os.path.join(opt['baseroot'], opt['lphase'], "IN_CR", img_k)):
                    if f.endswith('.png'):
                        in_path = os.path.join(opt['baseroot'], opt['lphase'], "IN_CR", img_k, f)
                        self.val_inputs.append((gt_path, in_path))

    def imread_norm(self, img_path):
        img = cv2.imread(img_path)
        if self.max_size is not None:
            th = (self.max_size // 32) * 32
            if not self.square:
                tw = (int(th * 4 / 3) // 32) * 32
            else:
                tw = th

            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        return img

    def __getitem__(self, index):
        scale = self.opt['scale']
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        if self.opt['phase'] == 'train':
            index = index % len(self.img_index.keys())
            gt_path = self.img_index[list(self.img_index.keys())[index]]['gt']
            if np.random.rand() > 0.9:
                lq_path = self.img_index[list(self.img_index.keys())[index]]['insh']
            else:
                k = np.random.randint(len(self.img_index[list(self.img_index.keys())[index]]['incr']))
                lq_path = self.img_index[list(self.img_index.keys())[index]]['incr'][k]
        else:
            gt_path, lq_path = self.val_inputs[index]
        try:
            img_gt = self.imread_norm(gt_path)
            img_lq = self.imread_norm(lq_path)
        except cv2.error as e:
            print(e)
            print(lq_path, gt_path)

        # augmentation for training
        if self.opt['phase'] == 'train':
            h, w, _ = img_gt.shape

            # padding
            img_gt, img_lq = padding(img_gt, img_lq, int(0.8 * h))

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, int(0.8 * h), scale,
                                                 gt_path)

            gt_size = self.opt['gt_size']
            ch, cw, _ = img_gt.shape
            if gt_size < ch:
                img_gt = cv2.resize(img_gt, (gt_size, gt_size), interpolation=cv2.INTER_LANCZOS4)
                img_lq = cv2.resize(img_lq, (gt_size, gt_size), interpolation=cv2.INTER_LANCZOS4)


            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        if self.opt['phase'] == 'train':
            return len(self.img_index.keys())
        else:
            return len(self.val_inputs)


if __name__ == '__main__':
    opt = {
        'baseroot': 'datasets/CL3AN',
        'lphase': 'Train',
        'phase': 'train',
        'geometric_augs': True,
        'scale': 1,
        'gt_size': 384,
        'maxsize': None,
    }

    dset = Dataset_PairedScene(opt)
    print(len(dset))
    batch = next(iter(dset))
    print(batch['gt'].size())
