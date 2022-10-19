import os
import gc
from datetime import datetime
import pickle
import h5py

import math
import numpy as np
import torch
import pytorch_kinematics as pk

import time
import pdb

### Code to create a reachability map using pytorch inverse kinematics (GPU-based tensor calculations)


# Use CUDA if available
if torch.cuda.is_available():
    d = "cuda"
    torch.cuda.empty_cache()
    print("[GPU MEMORY size in GiB]: " + str((torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_reserved(0))/1024**3))
else:
    d = "cpu"
dtype = torch.float32 # Choose float32 or 64 etc.


## Settings for the reachability map:
robot_urdf = "tiago_dual.urdf"
name_end_effector = "gripper_left_grasping_frame" # "arm_left_tool_link"
name_base_link = "base_footprint"
n_dof = 8 # Implied from the URDF and chosen links
use_torso = False
n_dof = 8 # Implied from the URDF and chosen links. 'use_torso=False' will reduce this by one in practice
# Number of DOFs and joint limits
joint_pos_min = torch.tensor([0.0, -1.1780972451, -1.1780972451, -0.785398163397, -0.392699081699, -2.09439510239, -1.41371669412, -2.09439510239], dtype=dtype, device=d)
joint_pos_max = torch.tensor([+0.35, +1.57079632679, +1.57079632679, +3.92699081699, +2.35619449019, +2.09439510239, +1.41371669412, +2.09439510239], dtype=dtype, device=d)
## Build kinematic chain from URDF
print("[Building kinematic chain from URDF...]:\n...\n...")
chain = pk.build_serial_chain_from_urdf(open(robot_urdf).read(), name_end_effector)
chain = chain.to(dtype=dtype, device=d)
assert (len(chain.get_joint_parameter_names()) == n_dof), "Incorrect number of DOFs set"
print("...\n...")
# Map resolution
angular_res = np.pi/8 # or 22.5 degrees per bin)
r_lim = [-np.pi, np.pi] # NOTE: Using 'intrinsic' euler rotations in XYZ
p_lim = [-np.pi/2, np.pi/2]
yaw_lim = [-np.pi, np.pi]
roll_bins = math.ceil((2*np.pi)/angular_res) # 16
pitch_bins = math.ceil((np.pi)/angular_res)  # 8. Only half the bins needed (half elevation and full azimuth sufficient to cover sphere)
yaw_bins = math.ceil((2*np.pi)/angular_res)  # 16
cartesian_res = 0.05 # metres
x_lim = [-1.2, 1.2] #[-1.0, 1.0] # min,max in metres (Set these as per your robot links)
y_lim = [-0.6, 1.35]#[-0.4, 1.15]
z_lim = [-0.35, 2.1]#[-0.15, 1.9]
x_bins = math.ceil((x_lim[1] - x_lim[0])/cartesian_res)
y_bins = math.ceil((y_lim[1] - y_lim[0])/cartesian_res)
z_bins = math.ceil((z_lim[1] - z_lim[0])/cartesian_res)
z_ind_offset = roll_bins*pitch_bins*yaw_bins
## IK settings:
ik_pos_error_thres = 0.025 # metres
ik_pos_sq_error_thresh = ik_pos_error_thres**2
ik_ang_error_thresh = 0.2617994 # 15 degrees
ik_ang_sq_error_thresh = ik_ang_error_thresh**2
MAX_ITER   = 500
DT         = torch.tensor([3e-1], dtype=dtype, device=d)
damp_coeff = torch.tensor([1e-12], dtype=dtype, device=d)
n_trials   = 15

## Create 6D reachability map tensor
num_values = 6+2 # Values in every voxel. Reachability map will store the 6D pose + two values: 'Visitation Frequency' and 'Manipulability'
# NOTE: For now we use an existing fk reach map and load the voxels
with open(os.getcwd()+'/../maps/gripper_left/filt_reach_map_gripper_left_grasping_frame_torso_False_0.05_2022-08-28-19-59-12.pkl','rb') as f:
    reach_map = pickle.load(f)
    nonzero_rows = np.abs(reach_map).sum(axis=1) > 0
    reach_map = reach_map[nonzero_rows] # Remove zero rows if they exist
    num_voxels = reach_map.shape[0]
    reach_map[:,6:8] = 0.0 # keep the voxel poses but set the scores to zeros
    reach_map = torch.tensor(reach_map, dtype=dtype)
    print("[Loaded existing FK reach map: filt_reach_map_gripper_left_grasping_frame_torso_False_0.05_2022-08-28-19-59-12.pkl]")
    print("[Number of 6D Voxels]: " + str(num_voxels))
# For speed, also create/load the inverse transforms of the voxels since we need these for error calculations
with open(os.getcwd()+'/../maps/gripper_left/inv_filt_reach_map_gripper_left_grasping_frame_torso_False_0.05_2022-08-28-19-59-12.pkl','rb') as f:
    inv_transfs = pickle.load(f)["inv_transf_batch"] # this is already a torch tensor
gc.collect()
# num_voxels = x_bins*y_bins*z_bins*roll_bins*pitch_bins*yaw_bins
# reach_map = np.zeros((num_voxels,num_values), dtype=dtype)
# print("[Number of 6D Voxels possible (Based on resolution settings)]: " + str(num_voxels))
# Full path and file name to save
reach_map_file_path = os.getcwd()+'/../maps/'
reach_map_file_name = 'IK_reach_map_'+str(name_end_effector)+'_torso_'+str(use_torso)+'_'+str(cartesian_res)+'_'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


## Create segments of the map (as per GPU memory limitations):
num_segments = 9 # NOTE: Tweak this paramter based on GPU Memory available
N_ik_loop = int(math.ceil(num_voxels/num_segments))
sampling_distr = torch.distributions.uniform.Uniform(joint_pos_min, joint_pos_max) # For sampling joints within limits

## Compute
print("[Starting IK calculations...]")
print("[Number of loops (map segments) is: "+str(num_segments)+"]")
print(f"[Number of IK iterations, retries is: {MAX_ITER},{n_trials}]")
print(f"[Will save file named: \'{reach_map_file_name}\' at path: \'{reach_map_file_path}\']")
print_time_est = True
t0 = time.perf_counter()
for seg in range(num_segments):
    # Get reach_map segment and get it's inverse transform for calculation
    inv_transfs_batch = inv_transfs[seg*N_ik_loop:(seg+1)*N_ik_loop].to(device=d)

    for trial in range(n_trials):        
        # Sample start joint config
        th_batch = sampling_distr.sample([N_ik_loop])
        if not use_torso:
            # Set torso joint pos to zero
            th_batch[:,0] = torch.tensor(0.0,dtype=dtype,device=d)
        for iter in range(MAX_ITER):
            loop_t0 = time.perf_counter()

            ee_transf_batch = chain.forward_kinematics(th_batch).get_matrix()
            torch.cuda.empty_cache() # Keep clearing cache to get rid of redundant variables
            transf_des_to_curr = torch.transpose(torch.bmm(inv_transfs_batch,ee_transf_batch),1,2) # batch of transfs (desired pose to current pose)
            log_err_batch = pk.transforms.se3.se3_log_map(transf_des_to_curr, eps=0.005) # se3_log function expects transposed transform
            
            # Calc Jacobian
            J = chain.jacobian(th_batch)
            if not use_torso:
                # Exclude torso joint jacobian
                J = J[:,:,1:] # Excluding first (torso) joint
            torch.cuda.empty_cache()
            
            # Calculate change in joint position: v = - J.T.dot(torch.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            pseudoInv_times_err = torch.linalg.solve(torch.bmm(J,torch.transpose(J,1,2)) + (damp_coeff * torch.eye(6,device=d)), log_err_batch)
            v = -torch.squeeze(torch.bmm(torch.transpose(J,1,2),torch.unsqueeze(pseudoInv_times_err,dim=2)))
            if not use_torso:
                # Add zero torso velocity
                v = torch.hstack(( torch.zeros((v.shape[0],1),dtype=dtype,device=d) , v ))
            th_batch = th_batch + (v*DT)
            # Clip joints to within joint limits
            th_batch = torch.max(torch.min(th_batch, joint_pos_max), joint_pos_min)
            del v, pseudoInv_times_err, transf_des_to_curr, ee_transf_batch
            torch.cuda.empty_cache()

            # Print loop computation time
            if(print_time_est):
                t_comp = time.perf_counter() - loop_t0
                print("IK iteration comp time = {0:.9e}s".format(t_comp))
                tot_hours = t_comp*MAX_ITER*n_trials*num_segments/3600
                print(f"Estimated total completion time is: {tot_hours} hours ")
                print_time_est = False

        # Check for IK success i.e. error is less than threshold
        success_mask = (log_err_batch[:,0:3].pow(2).sum(dim=1) < ik_pos_sq_error_thresh) & (log_err_batch[:,3:6].pow(2).sum(dim=1) < ik_ang_sq_error_thresh)
        
        # Calculate score (Manipulability) and copy to CPU
        M = torch.det(J[success_mask] @ torch.transpose(J[success_mask],1,2)).cpu() # Yoshikawa manipulability measure
        del th_batch, J, log_err_batch
        torch.cuda.empty_cache()
        
        # Add Visitation and (Average) Manipulability to Reachability Map
        reach_map_seg = reach_map[seg*N_ik_loop:(seg+1)*N_ik_loop]
        reach_map_seg[success_mask,-1] *= reach_map_seg[success_mask,-2] # Accumulate Manipulability. M*Visitation Frequency
        reach_map_seg[success_mask,-1] += M # Add the new M
        reach_map_seg[success_mask,-2] += 1 # Increment Visitation Frequency
        reach_map_seg[success_mask,-1] /= reach_map_seg[success_mask,-2] # divide to get the average
        torch.cuda.empty_cache()


# Save reachability map to file (as numpy pkl)
nonzero_scores = torch.abs(reach_map[:,6:]).sum(dim=1) > 0
reach_map_nonzero = reach_map[nonzero_scores].numpy()

with open(reach_map_file_path+reach_map_file_name+'.pkl', 'wb') as f:            
    pickle.dump(reach_map_nonzero,f) # Save only non-zero entries
    # pickle.dump(reach_map,f) # Optional: Save full map to add entries to it later

# Accumulate 6D voxel scores into 3D sphere scores (for visualization)
Manip_scaling = 500
indx = 0
first = True
while(indx < reach_map_nonzero.shape[0]):
    sphere_3d = reach_map_nonzero[indx][:3]
    # Count num_repetitions of current 3D sphere (in the next z_ind_offset subarray)
    num_repetitions = (reach_map_nonzero[indx:indx+z_ind_offset][:,:3] == sphere_3d).all(axis=1).sum().astype(dtype=np.int16)
    # Store sphere and average manipulability as the score. (Also, scale by a factor)    
    Manip_avg = reach_map_nonzero[indx:indx+num_repetitions, 7].mean()*Manip_scaling
    if first:
        first = False
        sphere_array = np.append(reach_map_nonzero[indx][:3], Manip_avg)
        # sphere_array = np.append(reach_map_nonzero[indx][:3], num_repetitions) # Optional: Use num_repetitions as score instead
        pose_array = np.append(reach_map_nonzero[0,:6], np.array([0., 0., 0., 1.])).astype(np.single) # dummy value
    else:
        sphere_array = np.vstack((sphere_array, np.append(reach_map_nonzero[indx][:3], Manip_avg)))
        # sphere_array = np.vstack((sphere_array, np.append(reach_map_nonzero[indx][:3], num_repetitions)))  # Optional: Use num_repetitions as score instead
        pose_array = np.vstack((pose_array, np.append(reach_map_nonzero[indx,:6], np.array([0., 0., 0., 1.])).astype(np.single))) # dummy value
    indx += num_repetitions
# Save 3D map as hdf5 (Mimic reuleux data structure)
with h5py.File(reach_map_file_path+"3D_"+reach_map_file_name+".h5", 'w') as f:
    sphereGroup = f.create_group('/Spheres')
    sphereDat = sphereGroup.create_dataset('sphere_dataset', data=sphere_array)
    sphereDat.attrs.create('Resolution', data=cartesian_res)
    # (Optional) Save all the 6D poses in each 3D sphere. Currently only dummy pose values (10 dimensional)
    poseGroup = f.create_group('/Poses')
    poseDat = poseGroup.create_dataset('poses_dataset', dtype=float, data=pose_array)


# END
t_comp = time.perf_counter() - t0
print("[TOTAL Comp Time] = {0:.2e}s".format(t_comp))

# Debug: Time perf counter
# pdb.set_trace()
# tmat = time.perf_counter()

# t_comp = time.perf_counter() - tmat
# print("Comp Time = {0:.9e}s".format(t_comp))
# pdb.set_trace()