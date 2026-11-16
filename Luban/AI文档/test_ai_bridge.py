import socket
import json
import base64
import time
import os
import shutil

class LubanLockClient:
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port

    def send_command(self, cmd_name, env_id=0, params=None):
        """发送 JSON 指令并接收响应"""
        payload = {
            "command": cmd_name,
            "env_id": env_id,
            "params": params or {}
        }
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5) 
                s.connect((self.host, self.port))
                
                message = json.dumps(payload) + '\n'
                s.sendall(message.encode('utf-8'))
                
                response_data = b""
                while True:
                    chunk = s.recv(8192 * 1024)
                    if not chunk:
                        break
                    response_data += chunk
                    if b'\n' in chunk:
                        break
                
                return json.loads(response_data.decode('utf-8'))
        except Exception as e:
            return {"status": "error", "result": str(e)}

def interactive_mode():
    client = LubanLockClient()
    print("=== 鲁班锁 AI 交互测试 (Bridge Mode) ===")
    print("可用快捷命令示例 (所有指令需首位指定 EnvID):")
    print("  move <env> <id> <axis> <dist>  (如: move 1 0 x 10 => 固定环境1的0号零件)")
    print("  push <env> <id> <axis> <dist>  (如: push 1 0 x 10 => 探测性移动)")
    print("  rotate <env> <id> <axis> <deg> (如: rotate 1 0 y 90)")
    print("  state <env>                    (获取指定环境状态)")
    print("  capture <env>                  (触发截图并返回路径)")
    print("  start <env> <level>            (重置指定环境的关卡)")
    print("  exit                           (退出)")
    print("-----------------------------------")

    while True:
        user_input = input(f"\nAI 命令 > ").strip().lower()
        if not user_input or user_input == 'exit':
            break

        parts = user_input.split()
        if len(parts) < 2:
            print("[!] 参数不足")
            continue

        cmd = parts[0]
        try:
            res = {}
            env_id = int(parts[1])
            if cmd == 'move' and len(parts) >= 5:
                # move 1 0 x 10
                res = client.send_command("move", env_id, {
                    "piece_id": int(parts[2]), 
                    "axis": parts[3], 
                    "distance": float(parts[4]),
                    "is_exploratory": False
                })

            elif cmd == 'push' and len(parts) >= 5:
                # push 1 0 x 10
                res = client.send_command("move", env_id, {
                    "piece_id": int(parts[2]), 
                    "axis": parts[3], 
                    "distance": float(parts[4]),
                    "is_exploratory": True
                })

            elif cmd == 'rotate' and len(parts) >= 5:
                # rotate 1 0 y 90
                res = client.send_command("rotate", env_id, {
                    "piece_id": int(parts[2]), 
                    "axis": parts[3], 
                    "angle": float(parts[4])
                })

            elif (cmd == 'state' or cmd == 'get_state') and len(parts) >= 2:
                # state 1
                res = client.send_command("get_state", env_id)
                data = res.get('data', {})
                if data:
                    print(f"[*] 关卡: {data.get('puzzle_index')}, 步数: {data.get('step_count')}, 完成: {data.get('is_solved')}")
                    pieces = data.get('pieces', [])
                    print(f"[*] 零件列表 ({len(pieces)}):")
                    for p in pieces[:5]: # 仅显示前 5 个
                        print(f"    - ID {p['id']}: Pos={p['pos']}, Rot={p['rot']}, Color={p['color']}, Finished={p['is_finished']}")

            elif cmd == 'capture' and len(parts) >= 2:
                # capture 1
                res = client.send_command("capture", env_id)
                data = res.get('data', {})
                path_str = data.get('screenshot_path', '')
                if path_str:
                    paths = path_str.split('|')
                    print(f"[*] 解析到截图: {len(paths)} 张")
                    for i, path in enumerate(paths):
                        if os.path.exists(path):
                            local_name = f"env_{env_id}_cam_{i}_{int(time.time())}.jpg"
                            shutil.copy(path, local_name)
                            print(f"    - 已保存副本: {local_name}")
                        else:
                            print(f"    - [!] 路径不存在: {path}")
                else:
                    print("[!] 未找到截图路径")

            elif cmd == 'start' and len(parts) >= 3:
                # start 1 0 (Env 1, Level 0)
                res = client.send_command("start_level", env_id, {
                    "level_index": int(parts[2])
                })
                print(f"[*] 已重置环境 {env_id} 至关卡 {parts[2]}")

            else:
                print("[!] 未知命令或参数格式错误")
                continue

            # Universal response printer
            status = res.get('status', 'n/a')
            if status == 'success':
                print(f">> SUCCESS")
            elif status in ['fail', 'error']:
                print(f">> FAILED: {res.get('result', 'Unknown error')}")

        except Exception as e:
            print(f"[!] 执行出错: {e}")

        except Exception as e:
            print(f"[!] 执行出错: {e}")

if __name__ == "__main__":
    interactive_mode()
