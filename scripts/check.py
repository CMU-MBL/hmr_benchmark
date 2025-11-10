import joblib

out_pkl_pth = 'output/WHAM/gymnasts/wham_output.pkl'
output = joblib.load(out_pkl_pth)
import pdb; pdb.set_trace()
print(f"Loaded {len(output)} frames from {out_pkl_pth}")