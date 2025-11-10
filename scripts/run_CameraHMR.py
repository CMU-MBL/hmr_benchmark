import argparse
import os
import sys
import torch
import cv2
import os.path as osp
import joblib
import logging as log

PROJ_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
CameraHMR_ROOT = os.path.join(PROJ_ROOT, "third_party", "CameraHMR")
sys.path.insert(0, CameraHMR_ROOT)
sys.path.insert(0, PROJ_ROOT)
os.chdir(CameraHMR_ROOT)

from third_party.CameraHMR.mesh_estimator import HumanMeshEstimator

_original_torch_load = torch.load 

def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load  

import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec



def make_parser():
    parser = argparse.ArgumentParser(description='CameraHMR Regressor')
    parser.add_argument("--video", type=str,  default='videos/gymnasts.mp4', 
        help="Path to input image folder.")
    parser.add_argument("--output_folder", type=str, default='output/CameraHMR',
        help="Path to folder output folder.")
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()

    args.video = osp.abspath(osp.join(PROJ_ROOT, args.video))
    args.output_folder = osp.abspath(osp.join(PROJ_ROOT, args.output_folder))

    estimator = HumanMeshEstimator()
    output = estimator.run_on_video(args.video, args.output_folder)
    out_pkl_pth = os.path.join(args.output_folder, "camerahmr_output.pkl")
    joblib.dump(dict(output), out_pkl_pth, compress=3)
    log.info(f"Save camerahmr output to {out_pkl_pth}")



if __name__=='__main__':
    main()

