"""
Multi-view rendering utilities for physics environments.

This module provides multi-view rendering capabilities to capture
the physics simulation from different camera angles and viewpoints.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
import math
from phyvpuzzle.core import CameraConfig

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

        self.camera_configs: Dict[str, CameraConfig] = {}
        self.set_camera_config(
            "front",
            position=[0, -1.5, 0.8],  # 正前方，y轴负方向看向原点
            target=[0, 0, 0.3],
            up=[0, 0, 1],
            fov=60.0,
            aspect_ratio=width / height,
            near_plane=0.1,
            far_plane=10.0,
            image_width=width,
            image_height=height
        )
        self.set_camera_config(
            "side",
            position=[1.5, 0, 0.8],   # 右侧，x轴正方向看向原点
            target=[0, 0, 0.3],
            up=[0, 0, 1],
            fov=60.0,
            aspect_ratio=width / height,
            near_plane=0.1,
            far_plane=10.0,
            image_width=width,
            image_height=height
        )
        self.set_camera_config(
            "top",
            position=[0, 0, 2.0],     # 顶部，z轴正方向俯视
            target=[0, 0, 0.3],
            up=[0, 1, 0],
            fov=60.0,
            aspect_ratio=width / height,
            near_plane=0.1,
            far_plane=10.0,
            image_width=width,
            image_height=height
        )
        self.set_camera_config(
            "perspective",
            position=[1.0, -1.0, 1.2],  # 斜上方视角
            target=[0, 0, 0.3],
            up=[0, 0, 1],
            fov=60.0,
            aspect_ratio=width / height,
            near_plane=0.1,
            far_plane=10.0,
            image_width=width,
            image_height=height
        )

    def set_camera_config(self, camera_name: str, position: List[float], 
                         target: List[float], up: List[float] = [0, 0, 1],
                         fov: float = 60.0, aspect_ratio: float = 1.0,
                         near_plane: float = 0.1, far_plane: float = 10.0,
                         image_width: Optional[int] = None, image_height: Optional[int] = None) -> None:
        """Set custom camera configuration using CameraConfig."""
        if image_width is None:
            image_width = self.width
        if image_height is None:
            image_height = self.height
        self.camera_configs[camera_name] = CameraConfig(
            position=tuple(position),
            target=tuple(target),
            up_vector=tuple(up),
            fov=fov,
            aspect_ratio=aspect_ratio,
            near_plane=near_plane,
            far_plane=far_plane,
            image_width=image_width,
            image_height=image_height
        )

    def render_single_view(self, camera_name: str,
                           labels: Optional[List[Tuple[List[float], str, Tuple[int, int, int]]]] = None
                           ) -> Image.Image:
        """Render from a single camera viewpoint."""
        if camera_name not in self.camera_configs:
            raise ValueError(f"Unknown camera: {camera_name}")

        config: CameraConfig = self.camera_configs[camera_name]

        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=list(config.position),
            cameraTargetPosition=list(config.target),
            cameraUpVector=list(config.up_vector)
        )

        # Compute projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=config.fov,
            aspect=config.aspect_ratio,
            nearVal=config.near_plane,
            farVal=config.far_plane
        )

        # Capture image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=config.image_width,
            height=config.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert to PIL Image.  getCameraImage returns a flat list when
        # PyBullet is built without NumPy, so reshape before slicing.
        rgb_array = np.reshape(np.asarray(rgb_img, dtype=np.uint8),
                               (height, width, -1))[:, :, :3]

        image = Image.fromarray(rgb_array)
        if labels:
            image = self._draw_object_labels(image, labels, view_matrix,
                                             projection_matrix)
        view = self._add_camera_label(image, camera_name)

        # Add label
        return view

    def render_multi_view(self, camera_order: List[str] = ["front", "side", "top", "perspective"],
                          labels: Optional[List[Tuple[List[float], str, Tuple[int, int, int]]]] = None
                          ) -> Image.Image:
        """Render multiple views and combine into single image."""
        views = {}

        for camera_name in camera_order:
            views[camera_name] = self.render_single_view(camera_name, labels=labels)

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

        # Thin separators so the four tiles read as separate views.
        draw = ImageDraw.Draw(combined_image)
        draw.line([(self.width, 0), (self.width, combined_height)],
                  fill=(60, 60, 60), width=3)
        draw.line([(0, self.height), (combined_width, self.height)],
                  fill=(60, 60, 60), width=3)
        return combined_image
        
    def render_circular_views(self, center: List[float] = [0, 0, 0.3], 
                            radius: float = 1.5, height: float = 0.8, 
                            num_views: int = 8) -> Dict[str, Image.Image]:
        """Render views arranged in a circle around the scene."""
        views = {}

        for i in range(num_views):
            angle = 2 * math.pi * i / num_views

            # Calculate camera position
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = height

            camera_name = f"circular_{i:02d}"

            self.set_camera_config(camera_name, [x, y, z], center)

            # Render view
            views[camera_name] = self.render_single_view(camera_name)

        return views

    def frame_to_bounds(self, aabb_min: List[float], aabb_max: List[float],
                        margin: float = 1.25, min_radius: float = 0.05) -> None:
        """Re-aim every camera at the given box and pull it in/out so the box
        fills the view, keeping each camera's viewing direction.

        The default cameras sit ~1.7 m from the origin, while puzzles are a
        few centimetres to ~0.5 m across, so without this the objects cover
        a few percent of each view.
        """
        lo = np.asarray(aabb_min, dtype=float)
        hi = np.asarray(aabb_max, dtype=float)
        center = (lo + hi) / 2
        radius = max(min_radius, float(np.linalg.norm(hi - lo)) / 2)
        for name, cfg in self.camera_configs.items():
            direction = (np.asarray(cfg.position, dtype=float)
                         - np.asarray(cfg.target, dtype=float))
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            half_fov = math.radians(cfg.fov) / 2
            dist = margin * radius / math.sin(half_fov)
            eye = center + direction / norm * dist
            self.set_camera_config(
                name, position=eye.tolist(), target=center.tolist(),
                up=list(cfg.up_vector), fov=cfg.fov,
                aspect_ratio=cfg.aspect_ratio,
                near_plane=max(0.01, dist - 3 * radius),
                far_plane=dist + 10 * radius + 5.0,
                image_width=cfg.image_width, image_height=cfg.image_height)

    def _draw_object_labels(self, image: Image.Image, labels, view_matrix,
                            projection_matrix) -> Image.Image:
        """Overlay object-id tags at the 2D projection of each 3D anchor.

        Agents address objects by id in tool calls; without on-image ids they
        have to guess which rendered piece an id refers to.
        """
        if not view_matrix or not projection_matrix:
            return image
        # PyBullet matrices are column-major flat lists.
        mvp = (np.asarray(projection_matrix, dtype=float).reshape(4, 4).T
               @ np.asarray(view_matrix, dtype=float).reshape(4, 4).T)
        out = image.copy()
        draw = ImageDraw.Draw(out)
        font = self._label_font(max(12, image.height // 34))
        w, h = image.size
        projected = []
        for point, text, color in labels:
            clip = mvp @ np.array([point[0], point[1], point[2], 1.0])
            if clip[3] <= 1e-6:
                continue  # behind the camera
            nx, ny = clip[0] / clip[3], clip[1] / clip[3]
            if abs(nx) > 1 or abs(ny) > 1:
                continue
            projected.append(((nx + 1) / 2 * w, (1 - ny) / 2 * h, text, color))

        # Greedy placement: nudge a tag up/down (alternating) until it no
        # longer overlaps a tag already placed; dense scenes (e.g. a
        # 15-piece lock) otherwise stack tags on top of each other.
        placed: List[Tuple[float, float, float, float]] = []

        def overlaps(r):
            return any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3]
                       for q in placed)

        for px, py, text, color in sorted(projected, key=lambda t: t[1]):
            box = draw.textbbox((0, 0), text, font=font)
            tw, th = box[2] - box[0], box[3] - box[1]
            bw, bh = tw + 8, th + 6
            x0, base_y = px - bw / 2, py - bh - 4
            x0 = min(max(x0, 0), w - bw)

            def inside(r):
                return r[1] >= 0 and r[3] <= h

            rect = (x0, base_y, x0 + bw, base_y + bh)
            for k in range(1, 25):
                if inside(rect) and not overlaps(rect):
                    break
                dy = (bh + 2) * ((k + 1) // 2) * (-1 if k % 2 else 1)
                rect = (x0, base_y + dy, x0 + bw, base_y + dy + bh)
            placed.append(rect)
            if abs(rect[1] - base_y) > 1:
                anchor_y = rect[3] if rect[1] < py else rect[1]
                draw.line([(px, py), (px, anchor_y)], fill=tuple(color), width=1)
            draw.rectangle(list(rect), fill="white", outline=tuple(color), width=2)
            draw.text((rect[0] + 4 - box[0], rect[1] + 3 - box[1]), text,
                      fill="black", font=font)
        return out

    @staticmethod
    def _label_font(size: int):
        for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()

    def _add_camera_label(self, image: Image.Image, label: str) -> Image.Image:
        """Add camera name label to image."""
        labeled_image = image.copy()
        draw = ImageDraw.Draw(labeled_image)
        font = self._label_font(max(14, image.height // 24))

        text = label.upper()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        padding = 5
        bg_bbox = [6, 6, 6 + text_width + 2 * padding,
                   6 + text_height + 2 * padding + 2]
        draw.rectangle(bg_bbox, fill=(30, 30, 30), outline="white")
        draw.text((6 + padding - text_bbox[0], 6 + padding - text_bbox[1]),
                  text, fill="white", font=font)
        return labeled_image
