"""
test_connection.py — Test RetroArch network command connection
Run this before deploy_ram.py to verify everything is set up correctly.
"""
import socket
import time

HOST = "127.0.0.1"
PORT = 55355


def send_cmd(sock, cmd):
    sock.sendto(cmd.encode(), (HOST, PORT))
    try:
        data, _ = sock.recvfrom(256)
        return data.decode().strip()
    except socket.timeout:
        return None


print("=" * 50)
print("RetroArch Connection Test")
print("=" * 50)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(1.0)

# Test 1 — basic connection
print("\n[1] Testing connection...")
resp = send_cmd(s, "VERSION")
if resp:
    print(f"    OK — RetroArch version: {resp}")
else:
    print("    FAIL — No response from RetroArch")
    print("    Fix: Settings > Network > Network Commands > ON")
    print("         Make sure RetroArch is open")
    s.close()
    input("\nPress Enter to exit...")
    exit(1)

# Test 2 — read Mario x position
print("\n[2] Reading Mario x position (RAM 0x006D)...")
resp = send_cmd(s, "READ_CORE_RAM 6D 1")
if resp:
    parts = resp.split()
    if len(parts) >= 3:
        val = int(parts[2], 16)
        print(f"    OK — x_pos low byte = {val}")
        if val == 0:
            print("    NOTE: value is 0 — make sure Mario ROM is loaded and game has started")
    else:
        print(f"    Unexpected response: {resp}")
else:
    print("    FAIL — Could not read RAM")
    print("    Fix: Make sure a NES core is loaded (Mesen or FCEUmm)")

# Test 3 — read world
print("\n[3] Reading current world (RAM 0x075C)...")
resp = send_cmd(s, "READ_CORE_RAM 75C 1")
if resp:
    parts = resp.split()
    if len(parts) >= 3:
        world = int(parts[2], 16) + 1
        print(f"    OK — World: {world}")
else:
    print("    FAIL")

# Test 4 — read multiple bytes (enemy positions)
print("\n[4] Reading enemy positions (RAM 0x0087, 5 bytes)...")
resp = send_cmd(s, "READ_CORE_RAM 87 5")
if resp:
    print(f"    OK — Enemy x positions: {resp}")
else:
    print("    FAIL")

# Test 5 — keyboard
print("\n[5] Testing pyautogui keyboard...")
try:
    import pyautogui
    print("    OK — pyautogui installed")
    print("    NOTE: Make sure RetroArch window is focused when deploy_ram.py runs")
except ImportError:
    print("    FAIL — pyautogui not installed")
    print("    Fix: pip install pyautogui")

s.close()

print("\n" + "=" * 50)
print("Summary:")
print("  - If all tests passed: run python deploy_ram.py")
print("  - Click RetroArch window BEFORE running deploy_ram.py")
print("  - Make sure game is unpaused and on the level screen")
print("=" * 50)
input("\nPress Enter to exit...")
