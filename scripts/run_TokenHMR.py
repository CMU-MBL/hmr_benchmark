import warnings
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra import main
# import argparse

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger

import torch
import os
import sys
import os.path as osp
import argparse


PROJ_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
TokenHMR_ROOT = os.path.join(PROJ_ROOT, "third_party", "TokenHMR")
sys.path.insert(0, TokenHMR_ROOT)
sys.path.insert(0, PROJ_ROOT)
os.chdir(TokenHMR_ROOT)

_original_torch_load = torch.load 

def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load  

import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class TokenHMRPredictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from third_party.TokenHMR.tokenhmr.lib.models import load_tokenhmr

        # Load checkpoints
        model, _ = load_tokenhmr(checkpoint_path=cfg.checkpoint, \
                                 model_cfg=cfg.model_config, \
                                 is_train_state=False, is_demo=True)

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)

        # Overriding the SMPL params with the TokenHMR params
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
            'pred_cam_t': model_out['pred_cam_t']
        }
        return out

class PHALP_Prime_TokenHMR(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = TokenHMRPredictor(self.cfg)

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, default='videos/gymnasts.mp4', 
                        help='input video path')

    parser.add_argument('--output_pth', type=str, default='output/TokenHMR', 
                        help='output folder to write results')

    args = parser.parse_args()

    args.video     = osp.abspath(osp.join(PROJ_ROOT, args.video))
    args.output_pth = osp.abspath(osp.join(PROJ_ROOT, args.output_pth))
    OmegaConf.set_struct(cfg, False)
    cfg.render.colors = 'slahmr'
    cfg.video.source = args.video
    cfg.output_dir = args.output_pth
    cfg.checkpoint = 'data/checkpoints/tokenhmr_model_latest.ckpt'
    cfg.model_config = 'data/checkpoints/model_config.yaml'
    OmegaConf.set_struct(cfg, True)

    """Main function for running the PHALP tracker."""
    phalp_tracker = PHALP_Prime_TokenHMR(cfg)

    from collections import defaultdict
    import joblib, numpy as np, os
    import cv2
    # phalp_tracker.track()
    final_visuals_dic, _ = phalp_tracker.track()
    output = defaultdict(lambda: {
        'pose': [],         # list of (72,)
        'trans': [],        # list of (3,)
        'betas': None,      # (B,) 
        'verts': [],        # list of (V,3)
        'frame_ids': []     # list of int
    })

    for frame_name, info in sorted(final_visuals_dic.items(), key=lambda kv: kv[1]['time']):
        t_idx = info['time']
        for i, tid in enumerate(info['tid']):
            smpl_params = info['smpl'][i]
            global_mat = np.squeeze(smpl_params['global_orient'])   # (3,3)
            rvec, _    = cv2.Rodrigues(global_mat)
            pose_root  = rvec.flatten()                             # (3,)
            body_mats  = np.squeeze(smpl_params['body_pose'])      # (J,3,3)
            pose_body  = []
            for mat in body_mats:
                rv, _ = cv2.Rodrigues(mat)
                pose_body.append(rv.flatten())
            pose_body = np.concatenate(pose_body, axis=0)           # (J*3,)
            pose = np.concatenate([pose_root, pose_body], axis=0) #(72,): 3 + J*3 = 72
            betas_np = smpl_params['betas']                     # (B,)
            trans = info['camera'][i]                           # (3,)
            device = torch.device("cuda:0")
            verts = phalp_tracker.HMAR.smpl(
                global_orient=torch.from_numpy(global_mat).float().unsqueeze(0).to(device),   # (B,3,3)
                body_pose=torch.from_numpy(body_mats).float().unsqueeze(0).to(device),        # (1,J,3,3)
                betas=torch.from_numpy(betas_np).float().unsqueeze(0).to(device),                # (1, num_betas)
                transl=torch.from_numpy(trans).float().unsqueeze(0).to(device)                # Bx3
            ).vertices.cpu().numpy()
            subj = output[tid]
            subj['pose'].append(pose)
            subj['trans'].append(trans)
            if subj['betas'] is None:
                subj['betas'] = betas_np
            subj['verts'].append(verts)
            subj['frame_ids'].append(t_idx)
    for tid, subj in output.items():
        subj['pose']        = np.stack(subj['pose'], axis=0)
        subj['trans']       = np.stack(subj['trans'], axis=0)
        subj['verts']       = np.stack(subj['verts'], axis=0)
        subj['frame_ids']   = np.array(subj['frame_ids'], dtype=int)
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_pth = os.path.join(cfg.output_dir, "tokenhmr_output.pkl")
    joblib.dump(dict(output), out_pth, compress=3)
    log.info(f"Save tokenhmr output to {out_pth}")

if __name__ == "__main__":
    main()
