import math
import numpy as np
import pytorch_kinematics as pk
import torch
import time
import pdb

if torch.cuda.is_available():
    d = "cuda"
    print("[GPU MEMORY available in GiB]: " + str((torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_reserved(0))/1024**3))
else:
    d = "cpu"
dtype = torch.float32

name_end_effector = "arm_left_tool_link"
# name_base_link = "torso_fixed_link"
name_base_link = "base_footprint"

# Joint-limits
n_dof = 8
joint_pos_min = np.array([0.0, -1.1780972451, -1.1780972451, -0.785398163397, -0.392699081699, -2.09439510239, -1.41371669412, -2.09439510239])
joint_pos_max = np.array([+0.35, +1.57079632679, +1.57079632679, +3.92699081699, +2.35619449019, +2.09439510239, +1.41371669412, +2.09439510239])

# chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
chain = pk.build_serial_chain_from_urdf(open("tiago_dual.urdf").read(), name_end_effector)
chain = chain.to(dtype=dtype, device=d)


num_loops = 2
N = int(5529600/num_loops) # 1000
torch.cuda.empty_cache()
th_batch = torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)

# TODO: Filter out joint configurations that lead to self-collision!
# Try pinnochio
# t0 = time.perf_counter()
# t_comp = time.perf_counter() - t0
# print("[Filtering Comp Time] = {0:.9e}s".format(t_comp))
# order of magnitudes faster when doing FK in parallel
# (N,4,4) transform matrix; only the one for the end effector is returned since end_only=True by default
pdb.set_trace()
t0 = time.perf_counter()
for i in range(num_loops):
    loop_t0 = time.perf_counter()
    
    tg_batch = chain.forward_kinematics(th_batch)
    # .detach().cpu().numpy()
    del tg_batch # delete after use to free up memory
    torch.cuda.empty_cache()

    J = chain.jacobian(th_batch)
    del J # delete after use to free up memory
    torch.cuda.empty_cache()

    t_comp = time.perf_counter() - loop_t0
    print("Loop: " + str(i) + ". Comp Time = {0:.9e}s".format(t_comp))
t_comp = time.perf_counter() - t0
print("[TOTAL Comp Time] = {0:.9e}s".format(t_comp))

# elapsed 8.44686508178711s for N=1000 when serial
# for i in range(N):
    # tg = chain.forward_kinematics(th_batch[i])