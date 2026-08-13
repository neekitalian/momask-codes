"""
Blender Script: Render SMPL Animations
======================================

Renders SMPL skeleton animations from PyTorch pose tensors using Blender.
Usage: blender --background --python render_smpl.py -- --poses poses.pt --output output.mp4

Author: Neekita Lian
Lab: Embodied Media Lab, KMD Keio
"""

import bpy
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Render SMPL animations in Blender')
    parser.add_argument('--poses', type=str, required=True, help='Path to poses.pt file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1280, 720], help='Resolution WxH')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--engine', type=str, default='EEVEE', choices=['EEVEE', 'CYCLES'], help='Render engine')
    parser.add_argument('--samples', type=int, default=128, help='Render samples (for Cycles)')
    parser.add_argument('--animation-start', type=int, default=0, help='Start frame')
    parser.add_argument('--animation-end', type=int, default=-1, help='End frame (-1 for all)')
    
    # Parse only known args (ignore Blender's args)
    args, unknown = parser.parse_known_args()
    return args

def load_poses(poses_path):
    """Load SMPL poses from PyTorch tensor."""
    print(f"[INFO] Loading poses from {poses_path}")
    data = torch.load(poses_path, map_location='cpu')
    poses = data['poses'].numpy()  # [T, 144] or [T, 72]
    print(f"[INFO] Loaded poses shape: {poses.shape}")
    return poses

def setup_blender_scene(resolution, fps, engine):
    """Setup Blender scene with rendering parameters."""
    print("[INFO] Setting up Blender scene...")
    
    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()
    
    # Scene settings
    scene = bpy.context.scene
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.fps = fps
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.image_settings.ffmpeg_codec = 'H264'
    scene.render.image_settings.ffmpeg_format = 'MPEG4'
    
    # Render engine
    if engine == 'CYCLES':
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 128
        scene.cycles.max_bounces = 12
    else:  # EEVEE
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = 64
    
    # Add camera
    bpy.ops.object.camera_add(location=(0, -5, 1.5))
    camera = bpy.context.active_object
    camera.rotation_euler = (np.pi/2.5, 0, 0)
    scene.camera = camera
    
    # Add lights
    bpy.ops.object.light_add(type='SUN', location=(3, 3, 3))
    sun = bpy.context.active_object
    sun.data.energy = 2.0
    
    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 2))
    area = bpy.context.active_object
    area.data.energy = 1.0
    area.data.size = 5.0
    
    print(f"[INFO] Scene setup complete (engine={engine}, {resolution[0]}x{resolution[1]}@{fps}fps)")

def create_smpl_skeleton(num_frames):
    """Create a simple skeleton armature from SMPL joints."""
    print(f"[INFO] Creating SMPL skeleton for {num_frames} frames...")
    
    # SMPL has 23 joints (+ 1 root = 24)
    # For simplicity, we'll create a stick figure representation
    
    bpy.ops.object.armature_add(location=(0, 0, 0))
    armature = bpy.context.active_object
    armature.name = 'SMPL_Armature'
    
    # Enter edit mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Clear default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    # Create parent bone (root)
    bone = armature.data.edit_bones.new('Root')
    bone.head = (0, 0, 0)
    bone.tail = (0, 0, 0.1)
    
    # Create simple hierarchy
    joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine', 'L_Knee', 'R_Knee',
        'Chest', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder',
        'Head', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
    ]
    
    parent_bone = bone
    for i, name in enumerate(joint_names):
        bone = armature.data.edit_bones.new(name)
        bone.head = parent_bone.tail
        bone.tail = bone.head + np.array([0, 0.1, 0])
        bone.parent = parent_bone
        if i % 2 == 0:
            parent_bone = bone
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Create mesh body
    verts = [(0, 0, 0), (0, 0, 1), (0.1, 0, 0.5), (-0.1, 0, 0.5)]
    faces = [(0, 1, 2), (0, 1, 3)]
    mesh = bpy.data.meshes.new('SMPL_Mesh')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    obj = bpy.data.objects.new('SMPL_Body', mesh)
    bpy.context.collection.objects.link(obj)
    
    # Add armature modifier
    mod = obj.modifiers.new(name='Armature', type='ARMATURE')
    mod.object = armature
    
    print(f"[INFO] Skeleton created with {len(armature.data.bones)} bones")
    return armature, obj

def animate_skeleton(armature, poses, fps):
    """Animate armature with pose data."""
    print(f"[INFO] Animating skeleton with {poses.shape[0]} frames...")
    
    num_frames = poses.shape[0]
    scene = bpy.context.scene
    
    # Set frame range
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    
    # For each frame, update bone rotations
    # This is simplified - full SMPL requires proper rotation encoding
    for frame_idx in range(num_frames):
        scene.frame_set(frame_idx)
        
        # Get pose for this frame
        pose = poses[frame_idx]
        
        # Update each bone (simplified rotation)
        for bone_idx, bone in enumerate(armature.pose.bones):
            if bone_idx < len(pose):
                # Convert pose params to rotation
                # This is a simplified version - full SMPL uses axis-angle
                rot_value = float(pose[bone_idx]) * 0.1
                bone.rotation_quaternion = (1, rot_value, rot_value, 0)
                bone.keyframe_insert(data_path='rotation_quaternion')
    
    print(f"[INFO] Animation complete ({num_frames} frames)")

def render_animation(output_path, scene):
    """Render the animation to video."""
    print(f"[INFO] Starting render to {output_path}...")
    
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(animation=True)
    
    print(f"[INFO] Render complete! Output: {output_path}")

def main():
    """Main rendering pipeline."""
    args = parse_args()
    
    print("=" * 70)
    print("  Unnoticed Dance - Blender SMPL Renderer")
    print("=" * 70)
    
    # Load poses
    poses = load_poses(args.poses)
    num_frames = poses.shape[0]
    
    # Setup scene
    setup_blender_scene(args.resolution, args.fps, args.engine)
    
    # Create skeleton
    armature, body = create_smpl_skeleton(num_frames)
    
    # Animate
    animate_skeleton(armature, poses, args.fps)
    
    # Render
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    scene = bpy.context.scene
    render_animation(output_path, scene)
    
    print("=" * 70)
    print("✓ Rendering complete!")
    print(f"  Output: {output_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
