"""
Multi-view rendering utilities for physics environments.

This module provides multi-view rendering capabilities to capture
the physics simulation from different camera angles and viewpoints.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
import math

# Conditional pybullet import
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    # Mock pybullet for testing
    class MockPyBullet:
        ER_BULLET_HARDWARE_OPENGL = 0
        @staticmethod
        def computeViewMatrix(*args, **kwargs): return []
        @staticmethod  
        def computeProjectionMatrixFOV(*args, **kwargs): return []
        @staticmethod
        def getCameraImage(*args, **kwargs): return (512, 512, np.zeros((512, 512, 3), dtype=np.uint8), None, None)
    p = MockPyBullet()


class MultiViewRenderer:
    """Renderer for capturing multiple viewpoints of the physics simulation."""
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        
        # Define default camera positions for multi-view rendering
        self.camera_configs = {
            "front": {
                "position": [0, -1.5, 0.8],
                "target": [0, 0, 0.3],
                "up": [0, 0, 1]
            },
            "side": {
                "position": [1.5, 0, 0.8], 
                "target": [0, 0, 0.3],
                "up": [0, 0, 1]
            },
            "top": {
                "position": [0, 0, 2.0],
                "target": [0, 0, 0.3],
                "up": [0, 1, 0]
            },
            "perspective": {
                "position": [1.0, -1.0, 1.2],
                "target": [0, 0, 0.3],
                "up": [0, 0, 1]
            }
        }
        
        # Camera parameters
        self.fov = 60
        self.near_plane = 0.1
        self.far_plane = 10.0
        
    def render_single_view(self, camera_name: str) -> Image.Image:
        """Render from a single camera viewpoint."""
        if camera_name not in self.camera_configs:
            raise ValueError(f"Unknown camera: {camera_name}")
            
        config = self.camera_configs[camera_name]
        
        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=config["position"],
            cameraTargetPosition=config["target"],
            cameraUpVector=config["up"]
        )
        
        # Compute projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=1.0,
            nearVal=self.near_plane,
            farVal=self.far_plane
        )
        
        # Capture image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to PIL Image
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        rgb_array = rgb_array.reshape(height, width, 3)
        
        image = Image.fromarray(rgb_array)
        
        # Add label
        return self._add_camera_label(image, camera_name)
    
    def render_multi_view(self, layout: str = "2x2") -> Image.Image:
        """Render multiple views and combine into single image."""
        if layout == "2x2":
            return self._render_2x2_layout()
        elif layout == "1x4":
            return self._render_1x4_layout()
        else:
            raise ValueError(f"Unsupported layout: {layout}")
    
    def _render_2x2_layout(self) -> Image.Image:
        """Render 2x2 grid layout with 4 views."""
        # Render individual views
        views = {}
        camera_order = ["front", "side", "top", "perspective"]
        
        for camera_name in camera_order:
            views[camera_name] = self.render_single_view(camera_name)
        
        # Create combined image
        combined_width = self.width * 2
        combined_height = self.height * 2
        combined_image = Image.new("RGB", (combined_width, combined_height), color="white")
        
        # Paste views in 2x2 grid
        positions = [
            (0, 0),                              # top-left: front
            (self.width, 0),                     # top-right: side  
            (0, self.height),                    # bottom-left: top
            (self.width, self.height)            # bottom-right: perspective
        ]
        
        for i, camera_name in enumerate(camera_order):
            combined_image.paste(views[camera_name], positions[i])
        
        return combined_image
    
    def _render_1x4_layout(self) -> Image.Image:
        """Render 1x4 horizontal layout."""
        # Render individual views
        views = {}
        camera_order = ["front", "side", "top", "perspective"]
        
        for camera_name in camera_order:
            views[camera_name] = self.render_single_view(camera_name)
        
        # Create combined image
        combined_width = self.width * 4
        combined_height = self.height
        combined_image = Image.new("RGB", (combined_width, combined_height), color="white")
        
        # Paste views horizontally
        for i, camera_name in enumerate(camera_order):
            x_pos = i * self.width
            combined_image.paste(views[camera_name], (x_pos, 0))
        
        return combined_image
    
    def _add_camera_label(self, image: Image.Image, label: str) -> Image.Image:
        """Add camera name label to image."""
        # Create a copy to avoid modifying original
        labeled_image = image.copy()
        draw = ImageDraw.Draw(labeled_image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Draw label background
        text_bbox = draw.textbbox((0, 0), label.upper(), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        padding = 4
        bg_bbox = [
            5, 5,
            5 + text_width + 2 * padding,
            5 + text_height + 2 * padding
        ]
        
        draw.rectangle(bg_bbox, fill="black", outline="white")
        
        # Draw text
        draw.text(
            (5 + padding, 5 + padding),
            label.upper(),
            fill="white",
            font=font
        )
        
        return labeled_image
    
    def render_custom_views(self, camera_configs: Dict[str, Dict[str, List[float]]]) -> Dict[str, Image.Image]:
        """Render from custom camera configurations."""
        views = {}
        
        for camera_name, config in camera_configs.items():
            # Compute view matrix
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=config["position"],
                cameraTargetPosition=config["target"],
                cameraUpVector=config["up"]
            )
            
            # Compute projection matrix
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=config.get("fov", self.fov),
                aspect=1.0,
                nearVal=config.get("near", self.near_plane),
                farVal=config.get("far", self.far_plane)
            )
            
            # Capture image
            width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Convert to PIL Image
            rgb_array = np.array(rgb_img, dtype=np.uint8)
            rgb_array = rgb_array[:, :, :3]
            rgb_array = rgb_array.reshape(height, width, 3)
            
            image = Image.fromarray(rgb_array)
            views[camera_name] = self._add_camera_label(image, camera_name)
        
        return views
    
    def set_camera_config(self, camera_name: str, position: List[float], 
                         target: List[float], up: List[float] = [0, 0, 1]) -> None:
        """Set custom camera configuration."""
        self.camera_configs[camera_name] = {
            "position": position,
            "target": target,
            "up": up
        }
    
    def create_circular_views(self, center: List[float] = [0, 0, 0.3], 
                            radius: float = 1.5, height: float = 0.8, 
                            num_views: int = 8) -> Dict[str, Image.Image]:
        """Create views arranged in a circle around the scene."""
        views = {}
        
        for i in range(num_views):
            angle = 2 * math.pi * i / num_views
            
            # Calculate camera position
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = height
            
            camera_name = f"circular_{i:02d}"
            
            # Temporarily set camera config
            original_config = self.camera_configs.get(camera_name)
            self.set_camera_config(camera_name, [x, y, z], center)
            
            # Render view
            views[camera_name] = self.render_single_view(camera_name)
            
            # Restore original config if it existed
            if original_config:
                self.camera_configs[camera_name] = original_config
            else:
                del self.camera_configs[camera_name]
        
        return views
