import os
import argparse
from datetime import datetime
import pickle
import h5py

import math
import numpy as np
import pytorch_kinematics as pk
import torch

import time
import pdb

### Code to invert a reachability map using pytorch kinematics (GPU-based tensor calculations)


# Use CUDA if available
if torch.cuda.is_available():
    d = "cuda"
    torch.cuda.empty_cache()
    print("[GPU MEMORY available in GiB]: " + str((torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_reserved(0))/1024**3))
else:
    d = "cpu"
dtype = torch.float32 # Choose float32 or 64 etc.


## 6D reachability map settings
angular_res = np.pi/8 # or 22.5 degrees per bin)
r_lim = [-np.pi, np.pi] # NOTE: Using 'intrinsic' euler rotations in XYZ
p_lim = [-np.pi/2, np.pi/2]
yaw_lim = [-np.pi, np.pi]
roll_bins = math.ceil((2*np.pi)/angular_res) # 16
pitch_bins = math.ceil((np.pi)/angular_res)  # 8. Only half the bins needed (half elevation and full azimuth sufficient to cover sphere)
yaw_bins = math.ceil((2*np.pi)/angular_res)  # 16
cartesian_res = 0.05 # metres

num_values = 6+2 # Values in every voxel. Reachability map has the 6D pose + two values: 'Visitation Frequency' and 'Manipulability'
# Full path and file name to save
parser = argparse.ArgumentParser("invert_reachability_map")
parser.add_argument("--map_pkl", type=str, required=True, help="Filename (with path) of the reachability map file to invert")
args, unknown = parser.parse_known_args()
reach_map_file_path = os.path.dirname(args.map_pkl)+'/'
reach_map_file_name = os.path.basename(args.map_pkl)
inv_reach_map_file_name = 'inv_' + reach_map_file_name


t0 = time.perf_counter()
## Load map
with open(args.map_pkl,'rb') as f:
    reach_map_filtered = pickle.load(f)
    nonzero_rows = np.abs(reach_map_filtered).sum(axis=1) > 0
    reach_map_filtered = reach_map_filtered[nonzero_rows] # Remove zero rows if they exist
    N_poses = reach_map_filtered.shape[0]

## create torch transforms
transf_batch = pk.transforms.Transform3d(pos=reach_map_filtered[:,:3], rot=reach_map_filtered[:,3:6], dtype=dtype, device=d) # NOTE: Using 'intrinsic' euler rotations in XYZ
torch.cuda.empty_cache()
## get inverse transforms and copy to CPU
inv_transf_batch = transf_batch._get_matrix_inverse()
del transf_batch
torch.cuda.empty_cache()

## Save inverse transforms and Manipulability scores to file (as torch pkl)
with open(reach_map_file_path+inv_reach_map_file_name,'wb') as f:
    save_dict = {"inv_transf_batch": inv_transf_batch.cpu(), "Vis_freq":torch.tensor(reach_map_filtered[:,6]), "M_scores":torch.tensor(reach_map_filtered[:,7])}
    pickle.dump(save_dict,f)
print(f"[Saved file: {inv_reach_map_file_name} ]")

## TODO: Create a 3D viz map
# poses_6d = torch.hstack((inv_transf_batch[:,:3,3], pk.transforms.matrix_to_euler_angles(inv_transf_batch[:,:3,:3], 'XYZ')))
# Discretize poses

# END
t_comp = time.perf_counter() - t0
print("[TOTAL Comp Time] = {0:.2e}s".format(t_comp))