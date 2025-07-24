# -*-coding:utf8-*-

import argparse
import os
from onedrivedownloader import download


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Download data')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    print('Downloading dataset')
    ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EYLmey_IMdVPtGCrCBx_CCMBToexGLjdFVy5mz5mo3Wpcg?download=1"
    download(ln, filename=os.path.join(args.data_path, 'miniImagenet.zip'),
             unzip=True, unzip_path=args.data_path, clean=True)
