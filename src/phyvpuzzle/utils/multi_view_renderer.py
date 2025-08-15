"""
Multi-view Rendering System

Provides multiple camera angles for better scene understanding.
"""

from typing import List, Dict, Tuple, Optional
from PIL import Image
import numpy as np
import pybullet as p


class MultiViewRenderer:
    """Renders the scene from multiple viewpoints."""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        
        # Define camera configurations for different views
        self.camera_configs = {
            'top': {
                'position': [0, 0, 2.5],
                'target': [0, 0, 0],
                'up_vector': [0, 1, 0],
                'fov': 60
            },
            'front': {
                'position': [2, 0, 0.8],
                'target': [0, 0, 0.3],
                'up_vector': [0, 0, 1],
                'fov': 60
            },
            'front_top': {
                'position': [2, -1, 1.5],
                'target': [0, 0, 0.3],
                'up_vector': [0, 0, 1],
                'fov': 60
            },
            'side': {
                'position': [0, 2, 0.8],
                'target': [0, 0, 0.3],
                'up_vector': [0, 0, 1],
                'fov': 60
            }
        }
    
    def render_single_view(self, view_name: str) -> Image.Image:
        """Render from a single viewpoint."""
        config = self.camera_configs.get(view_name, self.camera_configs['front_top'])
        
        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=config['position'],
            cameraTargetPosition=config['target'],
            cameraUpVector=config['up_vector']
        )
        
        # Compute projection matrix
        aspect_ratio = self.width / self.height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=config['fov'],
            aspect=aspect_ratio,
            nearVal=0.1,
            farVal=10.0
        )
        
        # Get camera image
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
        
        return Image.fromarray(rgb_array)
    
    def render_multi_view(self) -> Image.Image:
        """Render from multiple viewpoints and combine into a single image."""
        views = ['top', 'front', 'front_top', 'side']
        images = []
        
        for view in views:
            img = self.render_single_view(view)
            # Resize to half size for combining
            img = img.resize((self.width // 2, self.height // 2), Image.LANCZOS)
            images.append(img)
        
        # Create combined image (2x2 grid)
        combined = Image.new('RGB', (self.width, self.height))
        combined.paste(images[0], (0, 0))  # Top view - top left
        combined.paste(images[1], (self.width // 2, 0))  # Front view - top right
        combined.paste(images[2], (0, self.height // 2))  # Front-top view - bottom left
        combined.paste(images[3], (self.width // 2, self.height // 2))  # Side view - bottom right
        
        # Add labels to each view
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        
        # Try to use a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Add labels
        labels = ['TOP VIEW', 'FRONT VIEW', 'FRONT-TOP VIEW', 'SIDE VIEW']
        positions = [(10, 10), (self.width // 2 + 10, 10), 
                    (10, self.height // 2 + 10), (self.width // 2 + 10, self.height // 2 + 10)]
        
        for label, pos in zip(labels, positions):
            # Draw text with background for better visibility
            bbox = draw.textbbox(pos, label, font=font)
            draw.rectangle(bbox, fill='black')
            draw.text(pos, label, fill='white', font=font)
        
        return combined
    
    def get_best_view_for_action(self) -> Image.Image:
        """Get the best view for understanding the scene (front-top diagonal)."""
        return self.render_single_view('front_top')