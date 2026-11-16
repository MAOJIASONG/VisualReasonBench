import socket
import json
import time

class LuBanSuoClient:
    """
    鲁班锁 Unity 环境 AI 接口演示客户端
    封装常用的 Socket 请求，方便客户集成。
    """
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port

    def send_command(self, command, env_id=0, params=None):
        """发送 JSON 指令并获取返回结果"""
        payload = {
            "command": command,
            "env_id": env_id,
            "params": params or {}
        }
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10.0)
                s.connect((self.host, self.port))
                s.sendall(json.dumps(payload).encode('utf-8'))
                
                data = s.recv(1024 * 1024) # 1MB buffer for large states
                if not data:
                    return None
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            print(f"Error communicating with Unity: {e}")
            return None

def demo():
    client = LuBanSuoClient()
    target_env = 0

    print("=== Step 1: Start Level (开始第1关) ===")
    res = client.send_command("start_level", target_env, {"level_index": 1})
    print(f"Start Result: {res['status'] if res else 'Fail'}")

    time.sleep(1) # 等待加载

    print("\n=== Step 2: Get Initial State (获取初始状态) ===")
    res = client.send_command("get_state", target_env)
    if res and res['status'] == 'success':
        data = res['data']
        print(f"Level: {data['puzzle_index']}, Steps: {data['step_count']}")
        for piece in data['pieces'][:3]: # 仅显示前三个零件
            print(f"  Piece {piece['id']}: Pos={piece['pos']}, Rot={piece['rot']}, Color={piece['color']}")

    print("\n=== Step 3: Move Piece (移动零件 - 逻辑单位) ===")
    # 移动 ID 为 1 的零件，在 X 轴移动 10 个逻辑单位 (Unity中为 0.001)
    res = client.send_command("move", target_env, {
        "piece_id": 1, 
        "axis": "x", 
        "distance": 10.0,
        "is_exploratory": False
    })
    print(f"Move Success: {res['status'] if res else 'Fail'}")

    print("\n=== Step 4: Rotate Piece (旋转零件) ===")
    # 旋转 ID 为 1 的零件，绕 Y 轴旋转 90 度
    res = client.send_command("rotate", target_env, {
        "piece_id": 1, 
        "axis": "y", 
        "angle": 90.0
    })
    print(f"Rotate Success: {res['status'] if res else 'Fail'}")

    print("\n=== Step 5: Capture Screenshots (获取所有视角截图) ===")
    res = client.send_command("capture", target_env)
    if res and res['status'] == 'success':
        paths = res['data'].get('screenshot_path', '').split('|')
        print("Screenshots saved at:")
        for p in paths:
            print(f"  - {p}")

if __name__ == "__main__":
    demo()
