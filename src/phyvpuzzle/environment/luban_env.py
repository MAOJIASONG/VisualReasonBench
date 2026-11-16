"""
Luban Lock environment implementation backed by Unity.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import logging

from PIL import Image, ImageDraw

from phyvpuzzle.core import (
    BaseEnvironment,
    EnvironmentConfig,
    register_environment,
    register_environment_config,
)
from phyvpuzzle.core.base import Action, Observation, ObjectInfo, State
from phyvpuzzle.utils.multi_view_renderer import MultiViewRenderer

logger = logging.getLogger(__name__)


def _default_luban_exe() -> str:
    """Pick the Unity build binary for the current OS from the project's Luban/ dir.

    Linux  -> LuBanSuoEnv.x86_64 ; macOS -> the .app's MacOS binary ; else .exe.
    """
    luban_dir = Path(__file__).resolve().parents[3] / "Luban"
    if sys.platform.startswith("linux"):
        return str(luban_dir / "LuBanSuoEnv.x86_64")
    if sys.platform == "darwin":
        return str(luban_dir / "LuBanSuoEnv.app" / "Contents" / "MacOS" / "LuBanSuoEnv")
    return str(luban_dir / "LuBanSuoEnv.exe")


@register_environment_config("luban")
@dataclass
class LubanConfig(EnvironmentConfig):
    """Configuration for the Unity-backed Luban Lock environment."""

    host: str = "127.0.0.1"
    port: int = 9999
    env_id: int = 1
    level_index: int = 0
    action_mode: str = "native"
    exe_path: str = _default_luban_exe()
    start_level_on_reset: bool = True
    startup_wait: float = 1.5
    connect_timeout: float = 2.0
    request_timeout: float = 310.0
    max_startup_retries: int = 450
    startup_timeout: float = 90.0
    job_worker_count: int = 2
    logic_unit_scale: float = 0.0001
    # Keep legacy fields for config compatibility (unused with Unity backend).
    move_unit_step: float = 0.01
    rotate_unit_step: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        if not isinstance(self.port, int) or self.port <= 0:
            raise ValueError("port must be a positive integer")
        if not isinstance(self.env_id, int) or not (0 <= self.env_id <= 8):
            raise ValueError("env_id must be within [0, 8]")
        if not isinstance(self.level_index, int) or not (0 <= self.level_index <= 31):
            raise ValueError("level_index must be within [0, 31]")
        if self.action_mode not in {"native", "restored_original_rules"}:
            raise ValueError("action_mode must be native or restored_original_rules")
        if self.action_mode == "restored_original_rules" and self.env_id == 0:
            raise ValueError("Restored-rule mode requires env_id within [1, 8]")
        if self.startup_timeout <= 0 or self.request_timeout <= 0 or self.max_startup_retries <= 0:
            raise ValueError("Startup and request timeouts/retry counts must be positive")
        if type(self.job_worker_count) is not int or self.job_worker_count < 1:
            raise ValueError("job_worker_count must be a positive integer")
        if not os.path.isabs(self.exe_path):
            self.exe_path = str(Path(self.exe_path).resolve())


@register_environment("luban")
class LubanEnvironment(BaseEnvironment):
    """Unity-backed Luban Lock environment with socket control."""

    def __init__(self, config: LubanConfig):
        super().__init__(config)
        self.config: LubanConfig
        self.step_count: int = 0
        self.current_state: Optional[State] = None
        self.objects: List[ObjectInfo] = []
        self._last_state_data: Dict[str, Any] = {}
        self._process: Optional[subprocess.Popen] = None
        self._owns_process: bool = False
        self._server_ready: bool = False
        self._tool_handlers = {
            "get_state": self._tool_get_state,
            "move_piece": self._tool_move_piece,
            "rotate_piece": self._tool_rotate_piece,
        }
        if self._restored_mode:
            self._tool_handlers.update({
                "move_group": self._tool_move_group,
                "move_vector": self._tool_move_vector,
                "rotate_group": self._tool_rotate_group,
                "apply_special_rule": self._tool_apply_special_rule,
            })

    @property
    def _restored_mode(self) -> bool:
        return self.config.action_mode == "restored_original_rules"

    @staticmethod
    def _require_success(result: Dict[str, Any], operation: str) -> Dict[str, Any]:
        if result.get("status") != "success":
            raise RuntimeError(f"Luban {operation} failed: {result}")
        return result.get("data", {}) or {}

    # ------------------------------------------------------------------ #
    # BaseEnvironment API
    # ------------------------------------------------------------------ #
    def reset(self) -> Observation:
        """Reset environment to initial state."""
        self.step_count = 0
        self.current_state = None
        self.objects = []
        self._last_state_data = {}
        self._ensure_ready()
        if self.config.start_level_on_reset:
            command = "original_start" if self._restored_mode else "start_level"
            self._require_success(self._send_command(command, {"level_index": self.config.level_index}), command)
            if not self._restored_mode:
                time.sleep(self.config.startup_wait)
        self._refresh_state()
        self.current_state = self._get_current_state()
        return self._create_observation()

    def step(self, action: Action) -> Observation:
        """Execute an action (tool call) and return new observation."""
        self.step_count += 1
        tool_result = self.execute_tool_call(action.action_type, action.parameters)
        self._refresh_state()
        self.current_state = self._get_current_state(
            metadata={
                "tool_call": action.to_dict(),
                "tool_result": tool_result,
            }
        )
        return self._create_observation()

    def render(self, multi_view: bool = True) -> Union[Image.Image, Dict[str, Image.Image]]:
        """Render current environment state by requesting Unity screenshots."""
        return self._capture_and_compose()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return JSON schemas for tools exposed to the agent."""
        def build_schema(name: str, desc: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

        schemas = [
            build_schema(
                "get_state",
                "Fetch current state (positions, rotations, step count). Does NOT trigger screenshots.",
                {},
                [],
            ),
            build_schema(
                "move_piece",
                "Move a piece along an axis by a signed distance in puzzle logic units.",
                {
                    "piece_id": {"type": "integer", "description": "Target piece ID."},
                    "axis": {"type": "string", "enum": ["x", "y", "z"], "description": "Move axis."},
                    "distance": {"type": "number", "description": "Move distance in logic units."},
                    **({} if self._restored_mode else {"is_exploratory": {"type": "boolean", "default": False, "description": "Move until blocked if true."}}),
                },
                ["piece_id", "axis", "distance"],
            ),
            build_schema(
                "rotate_piece",
                "Rotate a piece by ±45 or ±90 degrees around an axis." if self._restored_mode
                else "Rotate a piece by 90-degree multiples around an axis.",
                {
                    "piece_id": {"type": "integer", "description": "Target piece ID."},
                    "axis": {"type": "string", "enum": ["x", "y", "z"], "description": "Rotation axis."},
                    "angle": ({"type": "integer", "enum": [-90, -45, 45, 90]} if self._restored_mode
                              else {"type": "number", "description": "Rotation angle in degrees (multiples of 90)."}),
                    **({"pivot": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                                  "description": "Optional pivot in puzzle logic units; defaults to this piece's origin."}}
                       if self._restored_mode else {}),
                },
                ["piece_id", "axis", "angle"],
            ),
        ]
        if self._restored_mode:
            ids = {"type": "array", "items": {"type": "integer"}, "minItems": 1, "uniqueItems": True}
            axis = {"type": "string", "enum": ["x", "y", "z"]}
            pivot = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
            schemas.extend([
                build_schema("move_group", "Translate a selected group with original collision checks.",
                             {"piece_ids": ids, "axis": axis, "distance": {"type": "number"}},
                             ["piece_ids", "axis", "distance"]),
                build_schema("move_vector", "Move pieces in a primitive integer direction. Displacement = vector × positive distance.",
                             {"piece_ids": ids, "vector": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                              "distance": {"type": "number", "exclusiveMinimum": 0}}, ["piece_ids", "vector", "distance"]),
                build_schema("rotate_group", "Rotate a group by ±45 or ±90 degrees; default pivot is the first piece's origin.",
                             {"piece_ids": ids, "axis": axis, "angle": {"type": "integer", "enum": [-90, -45, 45, 90]},
                              "pivot": pivot}, ["piece_ids", "axis", "angle"]),
                build_schema("apply_special_rule", "Apply a conditional combined move/rotation rule listed in applicable_special_rules by get_state.",
                             {"rule_index": {"type": "integer", "minimum": 0}}, ["rule_index"]),
            ])
        return schemas

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool calls."""
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return {"status": "error", "message": f"Unknown tool '{tool_name}'"}
        try:
            return handler(**arguments)
        except Exception as exc:
            return {"status": "error", "message": f"Tool '{tool_name}' failed: {exc}"}

    def close(self) -> None:
        """Clean up Unity process if started by this environment."""
        if self._process and self._owns_process:
            try:
                if os.name == "posix":
                    os.killpg(self._process.pid, signal.SIGTERM)
                else:
                    self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(self._process.pid, signal.SIGKILL)
                else:
                    self._process.kill()
                self._process.wait(timeout=5)
            except ProcessLookupError:
                pass
        self._process = None
        self._owns_process = False
        self._server_ready = False

    # ------------------------------------------------------------------ #
    # Unity communication helpers
    # ------------------------------------------------------------------ #
    def _ensure_ready(self) -> None:
        if self._server_ready:
            return
        if self._check_server_available():
            self._server_ready = True
            return
        self._start_unity_process()
        try:
            self._wait_for_server()
        except Exception:
            self.close()
            raise
        self._server_ready = True

    def _check_server_available(self) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config.connect_timeout)
                sock.connect((self.config.host, self.config.port))
            return True
        except Exception:
            return False

    def _start_unity_process(self) -> None:
        exe_path = self.config.exe_path
        if not os.path.exists(exe_path):
            raise FileNotFoundError(f"Unity Luban build not found at: {exe_path}")
        if self._process and self._process.poll() is None:
            return
        command = [exe_path, "-job-worker-count", str(self.config.job_worker_count)]
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            xvfb = shutil.which("xvfb-run")
            if xvfb is None:
                raise RuntimeError("Luban rendering requires a display; install xvfb or provide DISPLAY")
            command = [xvfb, "-a", "-s", "-screen 0 1280x720x24", *command]
        self._process = subprocess.Popen(
            command,
            cwd=str(Path(exe_path).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._owns_process = True

    def _wait_for_server(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout
        for _ in range(self.config.max_startup_retries):
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"Unity exited during startup ({self._process.returncode})")
            if self._check_server_available():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.2, remaining))
        raise RuntimeError("Unity server did not become ready in time.")

    def _send_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._ensure_ready()
        payload = {"command": command, "env_id": self.config.env_id, "params": params or {}}
        may_have_executed = False
        try:
            message = json.dumps(payload, allow_nan=False) + "\n"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config.request_timeout)
                sock.connect((self.config.host, self.config.port))
                may_have_executed = command in {"start_level", "reset", "move", "rotate", "original_start", "original_action"}
                sock.sendall(message.encode("utf-8"))
                response_data = b""
                while True:
                    chunk = sock.recv(1024 * 1024)
                    if not chunk:
                        break
                    response_data += chunk
                    if b"\n" in chunk:
                        break
            raw = response_data.split(b"\n")[0] if response_data else b""
            if not raw:
                return {"status": "error", "code": "empty_response", "execution_may_have_occurred": may_have_executed,
                        "message": "Empty response from Unity"}
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            return {"status": "error", "code": "communication_error",
                    "execution_may_have_occurred": may_have_executed,
                    "message": f"Unity communication failed: {exc}"}

    # ------------------------------------------------------------------ #
    # State handling
    # ------------------------------------------------------------------ #
    def _refresh_state(self) -> None:
        try:
            result = self._tool_get_state()
            data = self._require_success(result, "state refresh")
            if self._restored_mode:
                data = self._normalize_original_state(data)
        except Exception:
            # A failed read must not leave an earlier victory or pose in an observation.
            self._last_state_data = {}
            self.objects = []
            self.current_state = None
            raise
        self._last_state_data = data
        self.objects = self._build_objects_from_state(data)

    def _normalize_original_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get("mode") != "restored_original_rules":
            raise ValueError("Runtime did not return restored-rule state; prepare and select a repaired runtime")
        engine_pieces = data["engine_state"]["Pieces"]
        pieces = {p["Id"]: p for p in engine_pieces}
        native = data["native_positions"]
        if len(pieces) != len(engine_pieces) or len(native) != len(pieces) or {p["id"] for p in native} != set(pieces):
            raise ValueError("Native and restored piece IDs differ")
        finished = set(data["native_finished_piece_ids"])
        separated = self._scene_separated(native)
        result = dict(data)
        result.update({"step_count": self.step_count, "scene_separated": separated,
                       "is_solved": data.get("native_unity_is_solved") is True and separated,
                       "pieces": [{"id": p["id"], "pos": {axis: p[axis] for axis in "xyz"},
                                   "quaternion": p["visual_rotation"], "rot": pieces[p["id"]]["Orientation"],
                                   "half_rotate_axis": pieces[p["id"]]["HalfRotateAxis"],
                                   "is_finished": p["id"] in finished} for p in native]})
        return result

    @staticmethod
    def _scene_separated(pieces: List[Dict[str, Any]]) -> bool:
        # Use actual renderer/collider bounds; native finish flags alone can also be
        # triggered by translating an intact group into the finish volume.
        if len(pieces) < 2 or any("scene_world_bounds" not in p for p in pieces):
            return False
        bounds = [p["scene_world_bounds"] for p in pieces]
        for box in bounds:
            for axis in "xyz":
                low, high = box["min"][axis], box["max"][axis]
                if not math.isfinite(low) or not math.isfinite(high) or high <= low:
                    return False
        return all(any(max(a["min"][axis] - b["max"][axis], b["min"][axis] - a["max"][axis]) > 1e-5
                       for axis in "xyz") for i, a in enumerate(bounds) for b in bounds[i+1:])

    def _build_objects_from_state(self, data: Dict[str, Any]) -> List[ObjectInfo]:
        objects: List[ObjectInfo] = []
        for piece in data.get("pieces", []) or []:
            pos = piece.get("pos", {}) or {}
            obj_id = int(piece.get("id", -1))
            objects.append(
                ObjectInfo(
                    object_id=obj_id,
                    name=f"piece_{obj_id}",
                    position=(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))),
                    orientation=tuple(float(piece.get("quaternion", {}).get(k, 1.0 if k == "w" else 0.0)) for k in "xyzw"),
                    object_type="luban_piece",
                    properties={
                        "rot": piece.get("rot"),
                        "is_finished": piece.get("is_finished", False),
                        "half_rotate_axis": piece.get("half_rotate_axis", 0),
                    },
                )
            )
        return objects

    def _get_current_state(self, metadata: Optional[Dict[str, Any]] = None) -> State:
        meta: Dict[str, Any] = {
            "puzzle_index": self._last_state_data.get("puzzle_index"),
            "step_count": self._last_state_data.get("step_count"),
            "is_solved": self._last_state_data.get("is_solved"),
        }
        if metadata:
            meta.update(metadata)
        if self._restored_mode:
            meta.update({k: self._last_state_data.get(k) for k in
                         ["mode", "engine_state", "native_unity_is_solved", "scene_separated", "applicable_special_rules"]})
        return State(
            step=self.step_count,
            objects=self.objects,
            time_stamp=time.time(),
            metadata=meta,
        )

    def _create_observation(self) -> Observation:
        image = self._capture_and_compose()
        return Observation(image=image, state=self.current_state, description=self._get_state_description())

    def _get_state_description(self) -> str:
        if not self.current_state:
            return "Luban environment not initialized."
        meta = self.current_state.metadata or {}
        lines = [
            f"Luban Lock (env_id={self.config.env_id})",
            f"Puzzle index: {meta.get('puzzle_index')}, step_count: {meta.get('step_count')}, solved: {meta.get('is_solved')}",
        ]
        tool_call = meta.get("tool_call")
        tool_result = meta.get("tool_result")
        if tool_call and tool_result:
            lines.append(
                f"Last tool: {tool_call.get('action_type')} {tool_call.get('parameters')} -> "
                f"{tool_result.get('status')} ({tool_result.get('message')})"
            )
        if self.objects:
            lines.append(f"Pieces ({len(self.objects)}):")
            for obj in self.objects:
                rot = obj.properties.get("rot")
                pos = obj.position
                lines.append(
                    f"- id={obj.object_id}, pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), rot={rot}"
                    + (f", half_rotate_axis={obj.properties['half_rotate_axis']}, quaternion={obj.orientation}" if self._restored_mode else "")
                )
        if self._restored_mode:
            lines.append(f"Applicable special rules: {meta.get('applicable_special_rules', [])}")
        return "\n".join(lines)

    def _blank_image(self) -> Image.Image:
        return Image.new("RGB", (self.config.render_width, self.config.render_height), color="black")

    # Unity writes one screenshot per camera; these three give front, side
    # and top-down coverage of the lock.
    CAPTURE_VIEWS = (1, 3, 5)

    def _capture_and_compose(self) -> Image.Image:
        """Capture Unity screenshots and tile viewpoints 1/3/5 side by side.

        Missing or unreadable screenshots are logged (they used to become
        silent black tiles) and drawn as a labelled placeholder.  Tiles are
        scaled to a common height and labelled with a legible font.
        """
        result = self._send_command("capture")
        if self._restored_mode:
            self._require_success(result, "capture")
        if result.get("status") != "success":
            logger.warning("Luban capture failed: %s", result.get("message"))
            return self._blank_image()

        data = result.get("data", {}) or {}
        paths = [p for p in str(data.get("screenshot_path", "")).split("|") if p]
        if not paths:
            if self._restored_mode:
                raise RuntimeError("Luban capture returned no screenshot paths")
            logger.warning("Luban capture returned no screenshot paths")
            return self._blank_image()

        tiles: List[Image.Image] = []
        for idx in self.CAPTURE_VIEWS:
            img = None
            if idx < len(paths) and os.path.exists(paths[idx]):
                try:
                    img = Image.open(paths[idx]).convert("RGB")
                except Exception as exc:
                    logger.warning("Unreadable Luban screenshot %s: %s", paths[idx], exc)
            else:
                logger.warning("Luban screenshot for view %d missing (%d paths returned)",
                               idx, len(paths))
            tiles.append(img)

        real = [t for t in tiles if t is not None]
        if self._restored_mode and len(real) != len(self.CAPTURE_VIEWS):
            raise RuntimeError("Luban capture is incomplete; expected readable viewpoints 1, 3 and 5")
        size = ((real[0].width, max(t.height for t in real)) if real else
                (self.config.render_width, self.config.render_height))
        tiles = [t if t is not None else self._missing_tile(size) for t in tiles]
        height = max(t.height for t in tiles)
        tiles = [t if t.height == height else
                 t.resize((max(1, round(t.width * height / t.height)), height))
                 for t in tiles]
        gap = 4
        canvas = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), height),
                           color=(60, 60, 60))
        draw = ImageDraw.Draw(canvas)
        font = MultiViewRenderer._label_font(max(14, height // 24))
        x_offset = 0
        for view_idx, img in zip(self.CAPTURE_VIEWS, tiles):
            canvas.paste(img, (x_offset, 0))
            label = f"VIEW {view_idx}"
            box = draw.textbbox((0, 0), label, font=font)
            w, h = box[2] - box[0], box[3] - box[1]
            draw.rectangle([x_offset + 6, 6, x_offset + 16 + w, 16 + h],
                           fill=(30, 30, 30), outline="white")
            draw.text((x_offset + 11 - box[0], 11 - box[1]), label, fill="white", font=font)
            x_offset += img.width + gap
        return canvas

    def _missing_tile(self, size) -> Image.Image:
        tile = Image.new("RGB", size, color="black")
        font = MultiViewRenderer._label_font(max(14, size[1] // 20))
        ImageDraw.Draw(tile).text((12, size[1] // 2), "screenshot unavailable",
                                  fill=(220, 80, 80), font=font)
        return tile

    def _wait_until_stable(self) -> int:
        """Unity environment handles physics internally; no wait needed."""
        return 0

    def describe_objects(self) -> str:
        """Provide a summary for prompt history."""
        if not self.current_state:
            return "Luban environment not initialized."
        return self._get_state_description()

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #
    def _tool_get_state(self) -> Dict[str, Any]:
        return self._send_command("original_state" if self._restored_mode else "get_state")

    def _tool_move_piece(self, piece_id: int, axis: str, distance: float, is_exploratory: bool = False) -> Dict[str, Any]:
        if self._restored_mode:
            if is_exploratory:
                return {"status": "error", "message": "Restored mode checks the entire requested move; use an explicit distance."}
            return self._tool_move_group([piece_id], axis, distance)
        return self._send_command(
            "move",
            {
                "piece_id": int(piece_id),
                "axis": axis,
                "distance": float(distance),
                "is_exploratory": bool(is_exploratory),
            },
        )

    def _tool_rotate_piece(self, piece_id: int, axis: str, angle: float, pivot: Optional[List[float]] = None) -> Dict[str, Any]:
        if self._restored_mode:
            return self._tool_rotate_group([piece_id], axis, angle, pivot)
        if pivot is not None:
            return {"status": "error", "message": "Custom rotation pivots require restored-rule mode."}
        return self._send_command(
            "rotate",
            {
                "piece_id": int(piece_id),
                "axis": axis,
                "angle": float(angle),
            },
        )

    def _tool_move_group(self, piece_ids: List[int], axis: str, distance: float) -> Dict[str, Any]:
        return self._original_action({"kind": "move", "piece_ids": piece_ids,
                                     "axis": axis, "distance_exact": float(distance)})

    def _tool_move_vector(self, piece_ids: List[int], vector: List[int], distance: float) -> Dict[str, Any]:
        return self._original_action({"kind": "move_vector", "piece_ids": piece_ids,
                                     "vector": vector, "distance_exact": float(distance)})

    def _tool_rotate_group(self, piece_ids: List[int], axis: str, angle: float,
                           pivot: Optional[List[float]] = None) -> Dict[str, Any]:
        params = {"kind": "rotate", "piece_ids": piece_ids, "axis": axis, "angle": angle}
        if pivot is not None:
            params["pivot"] = pivot
        return self._original_action(params)

    def _tool_apply_special_rule(self, rule_index: int) -> Dict[str, Any]:
        return self._original_action({"kind": "special", "rule_index": rule_index})

    def _original_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._restored_mode:
            raise ValueError("This action requires restored-rule mode")
        if params["kind"] == "special":
            if type(params["rule_index"]) is not int or params["rule_index"] < 0:
                raise ValueError("rule_index must be a nonnegative integer")
        else:
            ids = params["piece_ids"]
            if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids)):
                raise ValueError("piece_ids must be distinct nonnegative integers")
            if "axis" in params and params["axis"] not in ("x", "y", "z"):
                raise ValueError("axis must be x, y or z")
            if "distance_exact" in params and not math.isfinite(params["distance_exact"]):
                raise ValueError("distance must be finite")
            if params["kind"] == "move_vector":
                vector = params["vector"]
                if not isinstance(vector, list) or len(vector) != 3 or any(type(i) is not int for i in vector) or math.gcd(*vector) != 1:
                    raise ValueError("vector must contain three primitive integer components")
                if params["distance_exact"] <= 0:
                    raise ValueError("vector distance must be positive")
            if params["kind"] == "rotate":
                if params["angle"] not in (-90, -45, 45, 90):
                    raise ValueError("rotation angle must be ±45 or ±90")
                if "pivot" in params and (not isinstance(params["pivot"], list) or len(params["pivot"]) != 3
                                           or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in params["pivot"])):
                    raise ValueError("pivot must contain three finite coordinates")
        return self._send_command("original_action", params)
