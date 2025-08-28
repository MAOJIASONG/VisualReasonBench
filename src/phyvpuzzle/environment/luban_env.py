"""
Luban lock environment implementation with constrained mode (world-fixed pieces).
"""

import math
from typing import Dict, List, Any
from .base_env import PhysicsEnvironment, p


class LubanEnvironment(PhysicsEnvironment):
    """Environment for Luban lock (wooden burr) puzzles with constrained mode."""

    def __init__(self, config):
        super().__init__(config)

        # Parse constrained mode configuration
        constrained_defaults = {
            "move_step_m": 0.01,
            "rotate_step_deg": 5.0,
            "max_force": 300.0,
            "erp": 0.4,
            "settle_steps": 8,
            "guard_contacts": True,
        }

        cfg = config.get("luban_constrained", {})
        self.luban_constrained_enabled = cfg.get("enabled", True)
        self.constrained_cfg = {**constrained_defaults, **cfg}

        # Validate and normalize critical configuration parameters
        cfg = self.constrained_cfg
        corrections_made = False

        # Validate step sizes
        if not isinstance(cfg["move_step_m"], (int, float)) or cfg["move_step_m"] <= 0:
            cfg["move_step_m"] = 0.01
            corrections_made = True
        if (
            not isinstance(cfg["rotate_step_deg"], (int, float))
            or cfg["rotate_step_deg"] <= 0
        ):
            cfg["rotate_step_deg"] = 5.0
            corrections_made = True

        # Validate physics parameters
        cfg["erp"] = max(0.0, min(1.0, float(cfg["erp"])))
        cfg["max_force"] = max(1e-3, float(cfg["max_force"]))
        cfg["settle_steps"] = max(1, int(cfg["settle_steps"]))

        # Initialize baseline constraints dictionary
        self.baseline_constraints: Dict[str, int] = {}

        # Check for verbose logging
        self.verbose_logging = config.get("verbose_logging", False)

        if corrections_made and self.verbose_logging:
            print(f"Warning: corrected invalid step size parameters to defaults")

        print(
            f"Luban environment initialized {'in constrained mode' if self.luban_constrained_enabled else 'in free-physics mode'}"
        )

    def _create_world_fixed_constraint(
        self, obj_id: int, pos: List[float], orn: List[float]
    ) -> int:
        """Create a world-fixed constraint for an object."""
        constraint_id = p.createConstraint(
            parentBodyUniqueId=obj_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=pos,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=orn,
            physicsClientId=self.client_id,
        )

        # Configure constraint parameters
        p.changeConstraint(
            constraint_id,
            maxForce=self.constrained_cfg["max_force"],
            erp=self.constrained_cfg["erp"],
            physicsClientId=self.client_id,
        )

        return constraint_id

    def _setup_task_environment(self):
        """Setup Luban lock specific environment."""
        # Check if we should use real URDF pieces or test pieces
        if self.config.get("use_real_luban_pieces", False):
            self._load_real_luban_pieces()
        else:
            self._create_test_luban_pieces()

        # Create baseline constraints only if constrained mode is enabled
        if self.luban_constrained_enabled:
            self._create_baseline_constraints()

    def _create_baseline_constraints(self):
        """Create world-fixed baseline constraints for all objects."""
        # Clear stale constraint IDs from any previous reset
        self.baseline_constraints.clear()
        constraint_count = 0

        for name, obj_info in self.objects.items():
            try:
                # Get current object position and orientation
                pos, orn = p.getBasePositionAndOrientation(
                    obj_info.object_id, physicsClientId=self.client_id
                )

                # Create world-fixed constraint
                constraint_id = self._create_world_fixed_constraint(
                    obj_info.object_id, list(pos), list(orn)
                )

                # Store constraint reference
                self.baseline_constraints[name] = constraint_id
                constraint_count += 1

                if self.verbose_logging:
                    print(
                        f"Created baseline constraint for {name} (constraint ID: {constraint_id})"
                    )

            except Exception as e:
                print(f"Failed to create baseline constraint for {name}: {e}")

        if self.verbose_logging:
            print(f"Created {constraint_count} baseline constraints for Luban pieces")

    def _create_test_luban_pieces(self):
        """Create simple test pieces to validate the holding system."""
        # Create 3 simple wooden block pieces for testing
        piece_configs = [
            {
                "name": "piece_1",
                "position": [0.0, 0.0, 0.5],
                "size": [0.08, 0.08, 0.02],
                "color": [0.6, 0.3, 0.1, 1],
            },
            {
                "name": "piece_2",
                "position": [0.1, 0.0, 0.5],
                "size": [0.08, 0.08, 0.02],
                "color": [0.7, 0.4, 0.2, 1],
            },
            {
                "name": "piece_3",
                "position": [-0.1, 0.0, 0.5],
                "size": [0.08, 0.08, 0.02],
                "color": [0.8, 0.5, 0.3, 1],
            },
        ]

        for piece_config in piece_configs:
            try:
                object_id = self.create_primitive_object(
                    object_name=piece_config["name"],
                    shape_type="box",
                    size=piece_config["size"],
                    position=piece_config["position"],
                    color=piece_config["color"],
                    mass=0.1,  # Light wooden pieces
                )
                print(f"Created test piece: {piece_config['name']} (ID: {object_id})")

            except Exception as e:
                print(f"Failed to create piece {piece_config['name']}: {e}")

    def _load_real_luban_pieces(self):
        """Load actual Luban URDF pieces from phobos_models."""
        import os

        # Path to Luban separate pieces
        luban_dir = self.urdf_base_path

        if not os.path.exists(luban_dir):
            print(f"Warning: Luban pieces directory not found: {luban_dir}")
            return

        # Scan directory for all available piece folders
        piece_names = []
        try:
            for item in os.listdir(luban_dir):
                item_path = os.path.join(luban_dir, item)
                if os.path.isdir(item_path) and item.startswith("obj_"):
                    # Check if the piece has a valid URDF file
                    urdf_file = os.path.join(item_path, "urdf", f"{item}.urdf")
                    if os.path.exists(urdf_file):
                        piece_names.append(item)

            # Sort pieces for consistent loading order
            piece_names.sort()
            print(f"Found {len(piece_names)} Luban pieces in directory: {piece_names}")

        except Exception as e:
            print(f"Error scanning Luban pieces directory: {e}")
            return

        if not piece_names:
            print("No valid Luban pieces found in directory")
            return

        # Determine how many pieces to load
        if self.config.get("load_all_pieces", False):
            pieces_to_load = piece_names
            print(f"Loading all {len(pieces_to_load)} available Luban pieces...")
        else:
            # Load first 4 pieces for testing
            pieces_to_load = piece_names[:4]
            print(f"Loading first {len(pieces_to_load)} Luban pieces for testing...")

        loaded_count = 0
        for piece_name in pieces_to_load:
            piece_dir = os.path.join(luban_dir, piece_name)
            urdf_file = os.path.join(piece_dir, "urdf", f"{piece_name}.urdf")

            try:
                object_id = self.add_object(
                    object_name=piece_name,
                    urdf_path=urdf_file,
                    position=[0, 0, 0],  # Let URDF define the position
                    orientation=[0, 0, 0, 1],
                    object_type="luban_piece",
                )
                # Sync stored pose to live Bullet pose (URDF may impose offsets)
                try:
                    live_pos, live_orn = p.getBasePositionAndOrientation(
                        object_id, physicsClientId=self.client_id
                    )
                    obj_info = self.objects.get(piece_name)
                    if obj_info is not None:
                        obj_info.position = tuple(live_pos)
                        obj_info.orientation = tuple(live_orn)
                except Exception:
                    pass
                print(f"Loaded Luban piece: {piece_name} (ID: {object_id})")
                loaded_count += 1

            except Exception as e:
                print(f"Failed to load {piece_name}: {e}")
                print(f"  URDF path: {urdf_file}")

        print(
            f"Successfully loaded {loaded_count}/{len(pieces_to_load)} Luban pieces from {luban_dir}"
        )

    def _evaluate_success(self) -> bool:
        """Evaluate if Luban lock is solved."""
        # Simple success criteria for testing: all pieces are within close proximity
        if len(self.objects) < 2:
            return False

        positions = []
        for obj_name in self.objects.keys():
            obj_state = self.get_object_state(obj_name)
            if obj_state:
                positions.append(obj_state["position"])

        if len(positions) < 2:
            return False

        # Check if all pieces are within 0.2 units of each other
        max_distance = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos1, pos2 = positions[i], positions[j]
                distance = (
                    (pos1[0] - pos2[0]) ** 2
                    + (pos1[1] - pos2[1]) ** 2
                    + (pos1[2] - pos2[2]) ** 2
                ) ** 0.5
                max_distance = max(max_distance, distance)

        success = max_distance < 0.2

        if success:
            print(
                f"Luban puzzle solved! Max distance between pieces: {max_distance:.3f}"
            )

        return success

    def _calculate_penetration_metric(self) -> float:
        """Calculate total penetration metric for guard_contacts checking."""
        contacts = p.getContactPoints(physicsClientId=self.client_id)
        penetration = sum(abs(contact[8]) for contact in contacts if contact[8] < 0)
        return penetration

    # Tool overrides for constrained mode
    def _tool_pick(
        self, object_id: str, reset_orientation: bool = False
    ) -> Dict[str, Any]:
        """Pick tool override for constrained mode."""
        if not self.luban_constrained_enabled:
            return super()._tool_pick(object_id, reset_orientation)

        if object_id not in self.objects:
            return {"status": "error", "message": f"Unknown object: {object_id}"}

        if object_id not in self.baseline_constraints:
            return {
                "status": "error",
                "message": f"Object {object_id} not in baseline constraints",
            }

        # Enforce single selection - auto-swap if different object is selected
        auto_swapped = False
        current = next(iter(self.picked_objects), None)
        if current and current != object_id:
            auto_swapped = True
            self.picked_objects.clear()

        # Add to picked objects if not already selected
        self.picked_objects.add(object_id)

        baseline_cid = self.baseline_constraints[object_id]

        # Sync baseline constraint to the live pose; optionally reset orientation
        obj_info = self.objects[object_id]
        try:
            current_pos, current_orn = p.getBasePositionAndOrientation(
                obj_info.object_id, physicsClientId=self.client_id
            )
        except Exception:
            current_pos, current_orn = (
                list(obj_info.position),
                list(obj_info.orientation),
            )

        target_orn = [0, 0, 0, 1] if reset_orientation else list(current_orn)
        if reset_orientation:
            obj_info.orientation = (0, 0, 0, 1)
        # Always update child pivot to the live position to avoid origin snaps
        p.changeConstraint(
            baseline_cid,
            jointChildPivot=list(current_pos),
            jointChildFrameOrientation=target_orn,
            maxForce=self.constrained_cfg["max_force"],
            erp=self.constrained_cfg["erp"],
            physicsClientId=self.client_id,
        )

        return {
            "status": "success",
            "message": f"Picked {object_id}"
            + (" (auto-swapped)" if auto_swapped else ""),
            "picked_objects": list(self.picked_objects),
            "constraint_id": baseline_cid,
            "auto_swapped": auto_swapped,
            "reset_orientation": reset_orientation,
        }

    def _tool_release(self, object_id: str) -> Dict[str, Any]:
        """Release tool override for constrained mode."""
        if not self.luban_constrained_enabled:
            return super()._tool_release(object_id)

        if object_id in self.picked_objects:
            self.picked_objects.remove(object_id)

        return {
            "status": "success",
            "message": f"Released {object_id} (baseline constraint remains active)",
            "picked_objects": list(self.picked_objects),
        }

    def _tool_move(
        self, object_id: str, position: List[float], disable_second_hand: bool = False
    ) -> Dict[str, Any]:
        """Move tool override for constrained mode with interpolation.

        Note: if guard_contacts aborts, intermediate pose is not rolled back.
        Callers should handle the error and re-plan.
        """
        if not self.luban_constrained_enabled:
            return super()._tool_move(object_id, position, disable_second_hand)

        if object_id not in self.picked_objects:
            return {
                "status": "error",
                "message": f"Object {object_id} must be picked before moving",
            }

        if len(position) != 3:
            return {"status": "error", "message": "Position must have 3 coordinates"}

        if object_id not in self.baseline_constraints:
            return {
                "status": "error",
                "message": f"Object {object_id} not in baseline constraints",
            }

        obj_info = self.objects[object_id]
        baseline_cid = self.baseline_constraints[object_id]
        cfg = self.constrained_cfg

        # Calculate interpolation steps from the live Bullet pose (not cached)
        try:
            live_pos, _ = p.getBasePositionAndOrientation(
                obj_info.object_id, physicsClientId=self.client_id
            )
            start_pos = list(live_pos)
        except Exception:
            start_pos = list(obj_info.position)
        target_pos = position
        distance = math.sqrt(sum((target_pos[i] - start_pos[i]) ** 2 for i in range(3)))
        steps = max(1, math.ceil(distance / cfg["move_step_m"]))

        # Initialize guard_contacts tracking if enabled
        if cfg["guard_contacts"]:
            prev_pen = self._calculate_penetration_metric()
            worsen_count = 0

        # Perform interpolated movement
        for step in range(1, steps + 1):
            frac = step / steps
            pos_i = [
                start_pos[i] + frac * (target_pos[i] - start_pos[i]) for i in range(3)
            ]

            # Update constraint - use movement_force if available, fallback to max_force
            movement_force = cfg.get("movement_force", cfg["max_force"])
            p.changeConstraint(
                baseline_cid,
                jointChildPivot=pos_i,
                maxForce=movement_force,
                erp=cfg["erp"],
                physicsClientId=self.client_id,
            )

            # Step simulation
            for _ in range(cfg["settle_steps"]):
                p.stepSimulation(physicsClientId=self.client_id)
                if self.gui:
                    import time

                    time.sleep(0.001)

            # Check guard_contacts if enabled
            if cfg["guard_contacts"]:
                current_pen = self._calculate_penetration_metric()
                if current_pen > prev_pen + 1e-5:
                    worsen_count += 1
                else:
                    worsen_count = 0

                if worsen_count >= 2:
                    return {
                        "status": "error",
                        "message": f"Movement aborted due to increasing penetration (step {step}/{steps})",
                    }
                prev_pen = current_pen

        # Update object info
        obj_info.position = tuple(position)

        # Auto-deselect after successful move
        self.picked_objects.remove(object_id)

        return {
            "status": "success",
            "message": f"Moved {object_id} to {position} and auto-released",
            "picked_objects": list(self.picked_objects),
        }

    def _tool_rotate(self, object_id: str, axis: str, angle: float) -> Dict[str, Any]:
        """Rotate tool override for constrained mode with interpolation.

        Note: if guard_contacts aborts, intermediate pose is not rolled back.
        Callers should handle the error and re-plan.
        """
        if not self.luban_constrained_enabled:
            return super()._tool_rotate(object_id, axis, angle)

        if object_id not in self.picked_objects:
            return {
                "status": "error",
                "message": f"Object {object_id} must be picked before rotating",
            }

        if axis not in ["x", "y", "z"]:
            return {
                "status": "error",
                "message": f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'",
            }

        if object_id not in self.baseline_constraints:
            return {
                "status": "error",
                "message": f"Object {object_id} not in baseline constraints",
            }

        obj_info = self.objects[object_id]
        baseline_cid = self.baseline_constraints[object_id]
        cfg = self.constrained_cfg

        # Get live pose and target euler angles
        try:
            current_pos, current_orn = p.getBasePositionAndOrientation(
                obj_info.object_id, physicsClientId=self.client_id
            )
        except Exception:
            current_pos, current_orn = (
                list(obj_info.position),
                list(obj_info.orientation),
            )

        current_euler = list(
            p.getEulerFromQuaternion(current_orn, physicsClientId=self.client_id)
        )
        target_euler = current_euler.copy()

        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        target_euler[axis_index] += angle

        # Calculate interpolation steps
        angle_deg = abs(angle) * 180 / math.pi
        steps = max(1, math.ceil(angle_deg / cfg["rotate_step_deg"]))

        # Initialize guard_contacts tracking if enabled
        if cfg["guard_contacts"]:
            prev_pen = self._calculate_penetration_metric()
            worsen_count = 0

        # Perform interpolated rotation
        for step in range(1, steps + 1):
            frac = step / steps
            euler_i = [
                current_euler[i] + frac * (target_euler[i] - current_euler[i])
                for i in range(3)
            ]
            quat_i = p.getQuaternionFromEuler(euler_i, physicsClientId=self.client_id)

            # Update constraint - use movement_force if available, fallback to max_force
            movement_force = cfg.get("movement_force", cfg["max_force"])
            p.changeConstraint(
                baseline_cid,
                jointChildFrameOrientation=quat_i,
                jointChildPivot=list(
                    current_pos
                ),  # Preserve pivot to avoid implicit resets
                maxForce=movement_force,
                erp=cfg["erp"],
                physicsClientId=self.client_id,
            )

            # Step simulation
            for _ in range(cfg["settle_steps"]):
                p.stepSimulation(physicsClientId=self.client_id)
                if self.gui:
                    import time

                    time.sleep(0.001)

            # Check guard_contacts if enabled
            if cfg["guard_contacts"]:
                current_pen = self._calculate_penetration_metric()
                if current_pen > prev_pen + 1e-5:
                    worsen_count += 1
                else:
                    worsen_count = 0

                if worsen_count >= 2:
                    return {
                        "status": "error",
                        "message": f"Rotation aborted due to increasing penetration (step {step}/{steps})",
                    }
                prev_pen = current_pen

        # Update object info from live Bullet state
        try:
            _, end_orn = p.getBasePositionAndOrientation(
                obj_info.object_id, physicsClientId=self.client_id
            )
            obj_info.orientation = tuple(end_orn)
        except Exception:
            obj_info.orientation = tuple(
                p.getQuaternionFromEuler(target_euler, physicsClientId=self.client_id)
            )

        # Auto-deselect after successful rotation
        self.picked_objects.remove(object_id)

        return {
            "status": "success",
            "message": f"Rotated {object_id} {angle:.3f} radians around {axis}-axis and auto-released",
            "picked_objects": list(self.picked_objects),
        }

    def _tool_push(
        self, object_id: str, force: float, direction: List[float]
    ) -> Dict[str, Any]:
        """Push tool override - disabled in constrained mode."""
        if not self.luban_constrained_enabled:
            return super()._tool_push(object_id, force, direction)

        return {
            "status": "error",
            "message": "push is disabled in Luban constrained mode; use pick+move/rotate",
        }
