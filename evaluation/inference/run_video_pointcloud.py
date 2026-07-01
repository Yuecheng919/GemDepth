import argparse
import numpy as np
import os
import torch
import cv2
import glob
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)  
if root_dir not in sys.path:
    sys.path.append(root_dir)
from model.gemdepth import GemDepth
from model.utils.dc_utils import read_video_frames

def frame_to_world_points(depth, frame, intrinsic, extrinsic):
    if len(depth.shape) == 3:
        depth = depth.squeeze()

    h, w = depth.shape[:2]
    frame = np.asarray(frame)
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    if frame.dtype != np.uint8:
        frame = (frame * 255 if frame.max() <= 1.0 else frame).astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    inverse_depth = depth.astype(np.float32)
    valid = np.isfinite(inverse_depth) & (inverse_depth > 1e-6)
    depth = np.zeros_like(inverse_depth, dtype=np.float32)
    depth[valid] = 1.0 / inverse_depth[valid]
    v, u = np.indices((h, w), dtype=np.float32)
    z = depth.astype(np.float32)
    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    cam_points = np.stack((x, y, z, np.ones_like(z)), axis=-1)[valid]

    cam_to_world = np.linalg.inv(extrinsic)
    world_points = (cam_points @ cam_to_world.T)[:, :3]
    colors = frame[valid].reshape(-1, 3)
    return world_points, colors

def save_ply(points, colors, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    points = points.astype(np.float32)
    vertices = np.column_stack((points, colors))
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])
    with open(output_path, 'w') as f:
        np.savetxt(f, vertices, fmt="%.6f %.6f %.6f %d %d %d", header=header, comments="")

def save_frame_pointclouds(frames, depths, extrinsics, intrinsics, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    total = min(len(frames), len(depths), len(extrinsics), len(intrinsics))
    for i in range(total):
        points, colors = frame_to_world_points(depths[i], frames[i], intrinsics[i], extrinsics[i])
        output_path = os.path.join(output_dir, f"{prefix}_{i:06d}.ply")
        save_ply(points, colors, output_path)
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  pointcloud progress: {i + 1}/{total}")
    print(f"✓ pointclouds have saved: {output_dir}")

def save_combined_video(frames, depths, output_path, fps=30, grayscale=False):
    if len(frames) == 0 or len(depths) == 0:
        return
    min_len = min(len(frames), len(depths))
    frames, depths = frames[:min_len], depths[:min_len]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    first_frame = np.array(frames[0])
    h_frame, w_frame = first_frame.shape[:2]
    total_width = w_frame
    first_depth = depths[0]
    h_d_raw, w_d_raw = first_depth.shape[:2]
    scale = w_frame / w_d_raw
    new_h_depth = int(h_d_raw * scale)
    total_height = h_frame + new_h_depth
    all_depths = np.concatenate([depth.flatten() for depth in depths])
    # d_min, d_max = all_depths.min(), all_depths.max()
    d_min, d_max = np.percentile(all_depths, 2), np.percentile(all_depths, 98)
    if d_max <= d_min: d_min, d_max = 0.0, 1.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
    for i, (frame, depth) in enumerate(zip(frames, depths)):
        frame_img = np.array(frame)
        if frame_img.dtype != np.uint8:
            frame_img = (frame_img * 255 if frame_img.max() <= 1.0 else frame_img).astype(np.uint8)
        if len(frame_img.shape) == 2: frame_img = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
        elif frame_img.shape[2] == 3: frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        if frame_img.shape[1] != w_frame or frame_img.shape[0] != h_frame:
            frame_img = cv2.resize(frame_img, (w_frame, h_frame))
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
        depth_uint8 = (np.clip(depth_norm, 0, 1) * 255).astype(np.uint8)
        if grayscale:
            depth_color = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        else:
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)        
        depth_color = cv2.resize(depth_color, (w_frame, new_h_depth))
        # Vertical Stack
        combined = np.vstack([frame_img, depth_color])   
        video_writer.write(combined)  
        if (i + 1) % 100 == 0:
            print(f"  progress: {i + 1}/{len(frames)}")   
    video_writer.release()
    print(f"✓ finish: {output_path}")

def save_depth_video(depths, output_path, fps=30, grayscale=False):

    if len(depths) == 0:
        print(f"error,depth empty")
        return   
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    first_depth = depths[0]
    if len(first_depth.shape) == 3 and first_depth.shape[2] == 1:
        first_depth = first_depth.squeeze()
    h, w = first_depth.shape[:2]   
    all_depths = np.concatenate([depth.flatten() for depth in depths])
    d_min, d_max = all_depths.min(), all_depths.max()
    if d_max <= d_min:
        d_min, d_max = 0.0, 1.0  
    print(f"  depth range: {d_min:.4f} - {d_max:.4f}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))    
    if not video_writer.isOpened():
        print(f"error: can't create {output_path}")
        return

    for i, depth in enumerate(depths):
        if len(depth.shape) == 3 and depth.shape[2] == 1:
            depth = depth.squeeze()
        depth_normalized = (depth - d_min) / (d_max - d_min + 1e-8)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        if grayscale:
            frame = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        video_writer.write(frame)
        if (i + 1) % 50 == 0 or (i + 1) == len(depths):
            print(f"  preogress: {i + 1}/{len(depths)}")
    video_writer.release()
    print(f"✓ video have saved: {output_path}")

def save_source_video(frames, output_path, fps=30):
    if len(frames) == 0:
        print(f"error,depth empty")
        return 
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    first_frame = frames[0]
    if isinstance(first_frame, np.ndarray):
        h, w = first_frame.shape[:2]
    else:
        first_frame_np = np.array(first_frame)
        h, w = first_frame_np.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))  
    if not video_writer.isOpened():
        print(f"error: can't create {output_path}")
        return
    
    for i, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)  
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)      
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)      
        video_writer.write(frame)
        
        if (i + 1) % 50 == 0 or (i + 1) == len(frames):
            print(f"  progress: {i + 1}/{len(frames)}")
    
    video_writer.release()
    print(f"✓ video have saved: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GemDepth Batch Processing')
    parser.add_argument('--input_dir', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--target_fps', type=int, default=-1)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--no_border', action='store_true')
    
    args = parser.parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    gemdepth = GemDepth(**model_configs[args.encoder])
    checkpoint = torch.load("./checkpoint/gemdepth.pth", map_location='cpu',weights_only=False)
    gemdepth.load_state_dict(checkpoint, strict=True)
    gemdepth = gemdepth.to(DEVICE).eval()
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    print(f"find {len(video_files)} video files to process")

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        current_output_dir = os.path.join(args.output_dir, base_name)
        os.makedirs(current_output_dir, exist_ok=True)
        print("\n" + "#"*60)
        print(f"processing: {video_name}")
        
        try:
            frames, target_fps = read_video_frames(video_path, args.max_len, args.target_fps, 1280)
            depths, extrinsics, intrinsics, fps = gemdepth.infer_video_geometry(
                frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32
            )
            processed_video_path = os.path.join(current_output_dir, base_name + '_src.mp4')
            depth_vis_path = os.path.join(current_output_dir, base_name + '_vis.mp4')
            combined_video_path = os.path.join(current_output_dir, base_name + '_combined.mp4')
            pointcloud_dir = os.path.join(current_output_dir, 'pointcloud')
            print(f"--- save result to: {current_output_dir} ---")
            save_source_video(frames, processed_video_path, fps=fps)
            save_depth_video(depths, depth_vis_path, fps=fps, grayscale=args.grayscale)
            save_combined_video(
                frames, depths, combined_video_path, 
                fps=fps, 
                grayscale=args.grayscale,
            )
            save_frame_pointclouds(frames, depths, extrinsics, intrinsics, pointcloud_dir, base_name)
        except Exception as e:
            print(f"process {video_name} error: {str(e)}")
            continue

    print("\n" + "="*60)
    print("finish all")