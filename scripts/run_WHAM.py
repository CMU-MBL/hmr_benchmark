import os, sys
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

PROJ_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
WHAM_ROOT = os.path.join(PROJ_ROOT, "third_party", "WHAM")
sys.path.insert(0, WHAM_ROOT)
sys.path.insert(0, PROJ_ROOT)
os.chdir(WHAM_ROOT)

from third_party.WHAM.configs.config import get_cfg_defaults
from third_party.WHAM.lib.data.datasets import CustomDataset
from third_party.WHAM.lib.utils.imutils import avg_preds
from third_party.WHAM.lib.utils.transforms import matrix_to_axis_angle
from third_party.WHAM.lib.models import build_network, build_body_model
from third_party.WHAM.lib.models.preproc.detector import DetectionModel
from third_party.WHAM.lib.models.preproc.extractor import FeatureExtractor
from third_party.WHAM.lib.models.smplify import TemporalSMPLify

import torch

_original_torch_load = torch.load 

def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load  

import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

try: 
    from third_party.WHAM.lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False


# no FLIP_EVAL, no Run Temporal SMPLify for post processing
def run(cfg,
        video,
        output_pth,
        network,
        run_global=False):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    # Preprocess
    with torch.no_grad():
        if not osp.exists(osp.join(output_pth, 'tracking_results.pth')):
            
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            slam = None
            
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break
                
                # 2D detection and tracking
                detector.track(img, fps, length)
                
                # SLAM
                if slam is not None: 
                    slam.track()
                
                bar.next()

            tracking_results = detector.process(fps)
            
            if slam is not None: 
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
        
            # Extract image features
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data
            # joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            logger.info(f'Save processed data at {output_pth}')
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            # slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run WHAM
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():           
            batch = dataset.load_data(subj)
            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
            
            # inference
            pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
    
        
        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam']).cpu().numpy()
        
        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id

    joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))
     
    # Visualize
    from third_party.WHAM.lib.vis.run_vis import run_vis_on_demo
    with torch.no_grad():
        run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, default='videos/gymnasts.mp4', 
                        help='input video path')

    parser.add_argument('--output_pth', type=str, default='output/WHAM', 
                        help='output folder to write results')

    args = parser.parse_args()

    args.video     = osp.abspath(osp.join(PROJ_ROOT, args.video))
    args.output_pth = osp.abspath(osp.join(PROJ_ROOT, args.output_pth))

    cfg = get_cfg_defaults()
    # cfg.merge_from_file('third_party/WHAM/configs/yamls/demo.yaml')
    cfg.merge_from_file(os.path.join(WHAM_ROOT, "configs", "yamls", "demo.yaml"))
    # cfg.DATASET.ROOT = os.path.join(WHAM_ROOT, "dataset")
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg, 
        args.video, 
        output_pth, 
        network, 
        run_global=False)
        
    print()
    logger.info('WHAM Done !')
