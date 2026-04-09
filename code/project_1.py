import pickle
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim


# 1. Setup & Initialization

# Change dataset ID here
DATASET_ID = "2" 

# Path 
data_root = "E:/2025 winter/276A/ECE276A_PR1/data" 

# Determine if training or testing based on ID
if DATASET_ID in ["10", "11"]: 
    imu_path = os.path.join(data_root, "testset/imu/imuRaw" + DATASET_ID + ".p")
    cam_path = os.path.join(data_root, "testset/cam/cam" + DATASET_ID + ".p")
    vic_path = None 
else:
    imu_path = os.path.join(data_root, "trainset/imu/imuRaw" + DATASET_ID + ".p")
    cam_path = os.path.join(data_root, "trainset/cam/cam" + DATASET_ID + ".p")
    vic_path = os.path.join(data_root, "trainset/vicon/viconRot" + DATASET_ID + ".p")

def read_data(fname):
    if fname is None or not os.path.exists(fname): return None
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        return None

def q2euler(q):
    w, x, y, z = q
    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi/2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def rot2euler(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy < 1e-6:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    else:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    return np.array([x, y, z])

def q_to_rot_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

# vicon_data to calculate  bias 
def convert_imu_data(imu_data, vicon_data=None):
    ts = imu_data[0, :]
    adc_accel = imu_data[1:4, :] 
    adc_gyro  = imu_data[4:7, :] 
    
    scale_accel = 3300.0 / 1023.0 / 330.0 * 9.81
    scale_gyro  = 3300.0 / 1023.0 / 3.33 * (np.pi / 180.0)
    
    # First 50 samples
    bias_gyro = np.mean(adc_gyro[:, :50], axis=1, keepdims=True)
    gyro_phys = (adc_gyro - bias_gyro) * scale_gyro
    
    # Bias Logic for Accelerometer
    mean_static_raw = np.mean(adc_accel[:, :50], axis=1)
    accel_raw_phys = mean_static_raw * scale_accel
    
    if vicon_data is not None:
        # Use Vicon GT at t=0 to find expected gravity component
        R0 = vicon_data['rots'][:, :, 0]
        g_world = np.array([0, 0, 9.81])
        g_body_expected = R0.T @ g_world 
        
        bias_accel_phys = accel_raw_phys - g_body_expected
        accel_phys = (adc_accel * scale_accel) - bias_accel_phys[:, np.newaxis]
    else:
        bias_accel = np.mean(adc_accel[:, :50], axis=1, keepdims=True)
        accel_phys = (adc_accel - bias_accel) * scale_accel
        accel_phys[2, :] += 9.81 
    
    return ts, accel_phys, gyro_phys



# 2. Main Processing

imud = read_data(imu_path)
vicd = read_data(vic_path)
camd = read_data(cam_path)

if imud is None:
    print(f"Error: Could not load IMU data from {imu_path}")
    sys.exit()

ts, accel_phys, gyro_phys = convert_imu_data(imud, vicd)
num_samples = ts.shape[0]

# Initialization
q_est = np.zeros((4, num_samples))
q_est[:, 0] = [1, 0, 0, 0]

# Dead Reckoning
for t in range(num_samples - 1):
    dt = ts[t+1] - ts[t]
    w = gyro_phys[:, t]
    angle = np.linalg.norm(w) * dt / 2.0
    axis = w / (np.linalg.norm(w) + 1e-8)
    dq = np.hstack([np.cos(angle), axis * np.sin(angle)])
    
    # Quaternion update
    qw, qx, qy, qz = q_est[:, t]
    dw, dx, dy, dz = dq
    q_est[:, t+1] = np.array([
        qw*dw - qx*dx - qy*dy - qz*dz,
        qw*dx + qx*dw + qy*dz - qz*dy,
        qw*dy - qx*dz + qy*dw + qz*dx,
        qw*dz + qx*dy - qy*dx + qz*dw
    ])

# Projected Gradient Descent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalize Accelerometer
accel_norm = np.linalg.norm(accel_phys, axis=0)
accel_norm[accel_norm < 1e-6] = 1.0
accel_tensor = torch.tensor((accel_phys / accel_norm).T, dtype=torch.float32, device=device)

q_tensor = torch.tensor(q_est.T, dtype=torch.float32, device=device, requires_grad=True)
gyro_tensor = torch.tensor(gyro_phys.T, dtype=torch.float32, device=device)
time_tensor = torch.tensor(ts, dtype=torch.float32, device=device)
g_ref = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0)

learning_rate = 0.1

def q_mult_batch(q, p):
    qw, qx, qy, qz = q[:,0], q[:,1], q[:,2], q[:,3]
    pw, px, py, pz = p[:,0], p[:,1], p[:,2], p[:,3]
    return torch.stack([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ], dim=1)

def rotate_vec_batch(q, v):
    q_vec = torch.cat([torch.zeros(v.shape[0], 1, device=device), v], dim=1)
    q_conj = torch.cat([q[:, 0:1], -q[:, 1:4]], dim=1)
    v_rot = q_mult_batch(q_mult_batch(q, q_vec), q_conj)
    return v_rot[:, 1:]

for epoch in range(1500): 
    if q_tensor.grad is not None:
        q_tensor.grad.zero_()
    
    # Motion Model Loss
    dt = (time_tensor[1:] - time_tensor[:-1]).unsqueeze(1)
    w = gyro_tensor[:-1]
    theta = torch.norm(w, dim=1, keepdim=True) * dt / 2.0
    w_dir = w / (torch.norm(w, dim=1, keepdim=True) + 1e-8)
    exp_w = torch.cat([torch.cos(theta), w_dir * torch.sin(theta)], dim=1)
    q_pred = q_mult_batch(q_tensor[:-1], exp_w)
    
    # Relative rotation 
    q_next_inv = q_tensor[1:].clone()
    q_next_inv[:, 1:] = -q_next_inv[:, 1:] 
    q_rel = q_mult_batch(q_next_inv, q_pred)
    
    # Log Map Loss:
    q_rel_w = q_rel[:, 0]
    q_rel_v = q_rel[:, 1:]
    v_norm = torch.norm(q_rel_v, dim=1)
    
    theta_err = 2 * torch.atan2(v_norm, torch.abs(q_rel_w))
    loss_motion = torch.mean(theta_err**2)
    
    # Observation Model Loss
    accel_world = rotate_vec_batch(q_tensor, accel_tensor)
    loss_obs = torch.mean((accel_world - g_ref)**2)
    
    # Total Loss
    loss = loss_motion + 0.1 * loss_obs
    loss.backward()
    
    # Update 
    with torch.no_grad():
        q_tensor -= learning_rate * q_tensor.grad
        q_tensor.div_(torch.norm(q_tensor, dim=1, keepdim=True))

q_opt = q_tensor.detach().cpu().numpy().T



# 3. Visualization

# Orientation Plot
euler_opt = np.array([q2euler(q) for q in q_opt.T]).T
euler_opt_deg = np.degrees(euler_opt) 

t_est = ts - ts[0]

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = ['Roll', 'Pitch', 'Yaw']

for i in range(3):
    axs[i].plot(t_est, euler_opt_deg[i, :], 'g-', linewidth=1.5, label='Optimized (IMU)')
    axs[i].set_ylabel(f'{labels[i]} (deg)')
    axs[i].grid(True)
    axs[i].set_ylim([-190, 190]) 
    axs[i].set_yticks(np.arange(-180, 181, 90))

if vicd is not None:
    vicon_rots = vicd['rots']
    vicon_ts = vicd['ts'].flatten()
    euler_gt = np.zeros((3, vicon_rots.shape[2]))
    for i in range(vicon_rots.shape[2]):
        euler_gt[:, i] = rot2euler(vicon_rots[:, :, i])

    euler_gt_deg = np.degrees(euler_gt)
    t_gt = vicon_ts - vicon_ts[0]
    for i in range(3):
        axs[i].plot(t_gt, euler_gt_deg[i, :], 'b--', linewidth=1.5, label='Truth (Vicon)')
        axs[i].legend(loc='upper right')

axs[0].set_title(f'Orientation Tracking Results (Dataset {DATASET_ID})')
axs[2].set_xlabel('Time (s)') 
plt.show()

# Panorama Stitching 
if camd is not None:
    cam_imgs = camd['cam']
    cam_ts = camd['ts'].flatten()
    H, W, _, K = cam_imgs.shape
    
    pan_h, pan_w = 500, 1000
    fov_h, fov_v = np.deg2rad(60), np.deg2rad(45)

    # Function to generate panorama from specific orientations
    def generate_panorama(orientations, timestamps_ori, title):
        panorama = np.zeros((pan_h, pan_w, 3))
        for k in range(0, K, 5): 
            # Find closest timestamp in the past
            past_idxs = np.where(timestamps_ori <= cam_ts[k])[0]
            if len(past_idxs) == 0: continue
            idx = past_idxs[-1]
            
            # Get rotation matrix (Estimated or GT)
            if orientations.ndim == 2: # q (4, N)
                q_curr = orientations[:, idx]
                R_bw = q_to_rot_matrix(q_curr)
            else: # R (3, 3, N)
                R_bw = orientations[:, :, idx]

            img = cam_imgs[:, :, :, k]
            
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            yp = -(u - W/2) / (W/2) * np.tan(fov_h/2)
            zp = -(v - H/2) / (H/2) * np.tan(fov_v/2)
            xp = np.ones_like(yp)
            
            vecs_body = np.stack([xp.flatten(), yp.flatten(), zp.flatten()])
            vecs_world = R_bw @ vecs_body
            
            # Spherical Mapping
            xw, yw, zw = vecs_world
            theta = np.arctan2(yw, xw)
            phi = np.arcsin(zw / np.linalg.norm(vecs_world, axis=0))
            
            px = ((theta + np.pi) / (2*np.pi) * pan_w).astype(int)
            py = ((-phi + np.pi/2) / np.pi * pan_h).astype(int)
            
            px = np.clip(px, 0, pan_w - 1)
            py = np.clip(py, 0, pan_h - 1)
            
            valid = (np.abs(phi) < np.pi/2.2)
            panorama[py[valid], px[valid], :] = img.reshape(-1, 3)[valid] / 255.0
        
        plt.figure(figsize=(12, 6))
        plt.imshow(panorama)
        plt.title(title)
        plt.axis('off')
        plt.show()

    # Estimated Panorama
    generate_panorama(q_opt, ts, f"Estimated Panorama (Dataset {DATASET_ID})")

    # Ground Truth Panorama 
    if vicd is not None:
        vicon_rots = vicd['rots']
        vicon_ts = vicd['ts'].flatten()
        generate_panorama(vicon_rots, vicon_ts, f"Ground Truth Panorama (Dataset {DATASET_ID})")
    
else:
    print("No Camera data available.")