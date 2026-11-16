"""Wrapper integration regressions. No Unity process or model API is started."""
import copy
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from phyvpuzzle.environment.luban_env import LubanConfig, LubanEnvironment


def original_state():
    return {
        "mode": "restored_original_rules", "puzzle_index": 0,
        "native_unity_is_solved": True, "engine_zone_estimate": True,
        "native_finished_piece_ids": [0], "applicable_special_rules": [1],
        "engine_state": {"Pieces": [{"Id": i, "Orientation": [0, 3, 5], "HalfRotateAxis": 1 if i == 0 else 0}
                                     for i in range(2)]},
        "native_positions": [
            {"id": i, "x": i * 3., "y": 0., "z": 0.,
             "visual_rotation": {"x": .38268343, "y": 0., "z": 0., "w": .92387953},
             "scene_world_bounds": {"min": {"x": i * 3., "y": 0., "z": 0.},
                                    "max": {"x": i * 3. + 1., "y": 1., "z": 1.}}}
            for i in range(2)],
    }


class EnvironmentIntegrationTests(unittest.TestCase):
    def environment(self, mode="restored_original_rules"):
        return LubanEnvironment(LubanConfig(type="luban", action_mode=mode))

    def test_native_default_preserves_tools_and_protocol(self):
        env = LubanEnvironment(LubanConfig(type="luban"))
        self.assertEqual({s["function"]["name"] for s in env.get_tool_schemas()},
                         {"get_state", "move_piece", "rotate_piece"})
        env._send_command = Mock(return_value={"status": "success"})
        env.execute_tool_call("rotate_piece", {"piece_id": 0, "axis": "x", "angle": 180})
        env._send_command.assert_called_once_with("rotate", {"piece_id": 0, "axis": "x", "angle": 180.})

    def test_original_tools_preserve_distance_and_reject_lossy_ids(self):
        env = self.environment()
        env._send_command = Mock(return_value={"status": "success"})
        distance = 10.1234567890123
        env.execute_tool_call("move_group", {"piece_ids": [0, 1], "axis": "z", "distance": distance})
        env._send_command.assert_called_once_with("original_action", {"kind": "move", "piece_ids": [0, 1],
                                                                      "axis": "z", "distance_exact": distance})
        env._send_command.reset_mock()
        for name, args in [("move_group", {"piece_ids": [1.5], "axis": "x", "distance": 1}),
                           ("move_vector", {"piece_ids": [0], "vector": [2, 2, 0], "distance": 1}),
                           ("apply_special_rule", {"rule_index": -1})]:
            self.assertEqual(env.execute_tool_call(name, args)["status"], "error")
        env._send_command.assert_not_called()

    def test_observation_uses_half_rotation_and_independent_scene_separation(self):
        env = self.environment()
        state = original_state()
        normalized = env._normalize_original_state(state)
        self.assertTrue(normalized["is_solved"])
        objects = env._build_objects_from_state(normalized)
        self.assertEqual(objects[0].properties["half_rotate_axis"], 1)
        self.assertAlmostEqual(objects[0].orientation[0], .38268343)
        state["native_positions"][1]["scene_world_bounds"] = copy.deepcopy(state["native_positions"][0]["scene_world_bounds"])
        self.assertFalse(env._normalize_original_state(state)["is_solved"], "Native win cannot certify an intact group")
        state = original_state()
        state["native_unity_is_solved"] = False
        self.assertFalse(env._normalize_original_state(state)["is_solved"], "Engine estimate cannot replace native victory")

    def test_failed_reset_and_state_read_never_return_stale_victory(self):
        env = self.environment()
        env._ensure_ready = Mock()
        env._send_command = Mock(return_value={"status": "error", "message": "load failed"})
        with self.assertRaises(RuntimeError):
            env.reset()
        env._send_command.assert_called_once_with("original_start", {"level_index": 0})
        env._last_state_data = {"is_solved": True}
        with self.assertRaises(RuntimeError):
            env._refresh_state()
        self.assertEqual(env._last_state_data, {})
        self.assertIsNone(env.current_state)

    def test_action_timeout_is_uncertain_and_is_not_retried(self):
        env = self.environment()
        env._ensure_ready = Mock()
        with patch("phyvpuzzle.environment.luban_env.socket.socket") as factory:
            sock = factory.return_value.__enter__.return_value
            sock.recv.side_effect = socket.timeout("late response")
            response = env._send_command("original_action", {"kind": "special", "rule_index": 1})
            self.assertEqual(response["status"], "error")
            self.assertTrue(response["execution_may_have_occurred"])
            sock.sendall.assert_called_once()

    def test_original_capture_failure_is_surfaced(self):
        env = self.environment()
        env._send_command = Mock(return_value={"status": "error", "message": "capture timeout"})
        with self.assertRaises(RuntimeError):
            env.render()

    def test_startup_can_wait_for_the_observed_thirteen_second_launch(self):
        env = self.environment()
        env._check_server_available = Mock(side_effect=[False] * 65 + [True])
        with patch("phyvpuzzle.environment.luban_env.time.monotonic", side_effect=[i / 5 for i in range(100)]), \
             patch("phyvpuzzle.environment.luban_env.time.sleep"):
            env._wait_for_server()
        self.assertEqual(env._check_server_available.call_count, 66)


if __name__ == "__main__":
    unittest.main()
