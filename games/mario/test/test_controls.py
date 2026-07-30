"""
test_controls.py — Comprehensive RetroArch + controls diagnostic
"""
import socket
import time
import subprocess
import os

HOST = "127.0.0.1"
PORT = 55355


def read_ram(sock, address):
    cmd = f"READ_CORE_RAM {address:X} 1\n".encode()
    sock.sendto(cmd, (HOST, PORT))
    try:
        data, _ = sock.recvfrom(64)
        parts = data.decode().strip().split()
        return int(parts[2], 16) if len(parts) >= 3 else 0
    except Exception:
        return 0


def get_x(sock):
    low    = read_ram(sock, 0x006D)  # x within screen
    screen = read_ram(sock, 0x006E)  # screen number
    return screen * 256 + low


print("=" * 55)
print("RetroArch Full Diagnostic")
print("=" * 55)

# ── Test 1: Process detection ─────────────────────────────
print("\n[1] Checking if RetroArch process is running...")
try:
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq retroarch.exe", "/FO", "CSV"],
        capture_output=True, text=True
    )
    if "retroarch.exe" in result.stdout.lower():
        print("    OK — retroarch.exe is running")
        # Get PID
        lines = [l for l in result.stdout.split('\n') if 'retroarch' in l.lower()]
        for l in lines:
            parts = l.strip('"').split('","')
            if len(parts) >= 2:
                print(f"       PID: {parts[1]}")
    else:
        print("    FAIL — retroarch.exe not found in running processes")
        print("    Fix: Open RetroArch first")
except Exception as e:
    print(f"    ERROR: {e}")

# ── Test 2: Window detection ──────────────────────────────
print("\n[2] Checking RetroArch window...")
try:
    import pygetwindow as gw
    wins = gw.getWindowsWithTitle("RetroArch")
    if wins:
        for w in wins:
            print(f"    OK — Found window: '{w.title}'")
            print(f"       Position: ({w.left}, {w.top})  Size: {w.width}x{w.height}")
            print(f"       Is active (focused): {w.isActive}")
            if not w.isActive:
                print("    NOTE: RetroArch is NOT the active window — activating it...")
                try:
                    w.activate()
                    time.sleep(0.5)
                    print("       Activated.")
                except Exception as e:
                    print(f"       Could not activate: {e}")
    else:
        print("    FAIL — No window with 'RetroArch' in title found")
        print("    (RetroArch may be running but titled differently)")
except ImportError:
    print("    SKIP — pygetwindow not available")
except Exception as e:
    print(f"    ERROR: {e}")

# ── Test 3: Network connection ────────────────────────────
print("\n[3] Testing RetroArch network commands...")
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(1.0)

s.sendto(b"VERSION\n", (HOST, PORT))
try:
    data, _ = s.recvfrom(64)
    print(f"    OK — RetroArch version: {data.decode().strip()}")
except socket.timeout:
    print("    FAIL — No response on port 55355")
    print("    Fix: Settings > Network > Network Commands > ON")
    s.close()
    input("\nPress Enter to exit...")
    exit(1)

# ── Test 4: Game state ────────────────────────────────────
print("\n[4] Reading game state...")
game_mode = read_ram(s, 0x0770)
x_pos     = get_x(s)
y_pos     = read_ram(s, 0x00CE)
world     = read_ram(s, 0x075C) + 1
stage     = read_ram(s, 0x075E) + 1
mode_str  = {0: "title screen", 1: "playing", 2: "dying/dead", 3: "level end"}.get(game_mode, f"unknown ({game_mode})")

print(f"    Game mode : {game_mode} = {mode_str}")
print(f"    Position  : x={x_pos}, y={y_pos}")
print(f"    World     : {world}-{stage}")

if game_mode != 1:
    print(f"\n    WARNING: Game is not in play mode (mode={game_mode})")
    print("    Fix: Get past title screen — press Enter in RetroArch to start")

# ── Test 5: pyautogui ─────────────────────────────────────
print("\n[5] Testing pyautogui...")
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    mx, my = pyautogui.position()
    print(f"    OK — pyautogui working, mouse at ({mx}, {my})")
except Exception as e:
    print(f"    FAIL — {e}")

# ── Test 6: Focus RetroArch and send keys ─────────────────
print("\n[6] Key input test...")
print("    Will auto-focus RetroArch window and send RIGHT key for 2 seconds")
input("    Press Enter to start (keep hands off keyboard)...")

# Auto-focus RetroArch
focused = False
try:
    import pygetwindow as gw
    wins = gw.getWindowsWithTitle("RetroArch")
    if wins:
        wins[0].activate()
        time.sleep(0.5)
        focused = True
        print("    Auto-focused RetroArch window")
except Exception:
    pass

if not focused:
    print("    Could not auto-focus — sending keys anyway in 2 seconds")
    print("    Click RetroArch window NOW!")
    time.sleep(2)

x_before = get_x(s)
print(f"    x before: {x_before}")
print("    Sending RIGHT for 2 seconds...")

import pyautogui
pyautogui.keyDown("right")
time.sleep(2.0)
pyautogui.keyUp("right")

x_after = get_x(s)
print(f"    x after:  {x_after}")

if x_after > x_before:
    print(f"\n    SUCCESS! Mario moved {x_after - x_before}px right — controls working!")
else:
    print(f"\n    FAIL — x did not change (before={x_before}, after={x_after})")
    print()
    print("    Diagnosis:")
    if game_mode != 1:
        print("    -> Game not in play mode — start the game first")
    else:
        print("    -> Game is in play mode but keys not reaching it")
        print("    -> Try: In RetroArch go to Settings > Input > Port 1 Controls")
        print("            Check what 'D-Pad Right' is mapped to")
        print("            Also check if 'Game Focus Toggle' is blocking input")
        print("            Try pressing Scroll Lock in RetroArch (toggles game focus)")

# ── Test 7: Alternative — send via RetroArch network cmd ─
print("\n[7] Testing alternative: JOYPAD_PRESSED via network...")
s.sendto(b"JOYPAD_PRESSED\n", (HOST, PORT))
try:
    resp = s.recvfrom(64)
    print(f"    Response: {resp}")
except Exception:
    print("    No response (expected)")

print("\n[8] Raw RAM dump around x position...")
# Read raw bytes to find correct x address
addrs = {
    "0x006D (x low)":   0x006D,
    "0x0086 (x page)":  0x0086,
    "0x03AD (x world)": 0x03AD,
    "0x00B5 (x scroll)":0x00B5,
    "0x006E (x screen)":0x006E,
}
s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s2.settimeout(0.5)
for name, addr in addrs.items():
    val = read_ram(s2, addr)
    print(f"    {name} = {val} (0x{val:02X})")

# Also read 16 bytes from 0x0080 to see the area
s2.sendto(b"READ_CORE_RAM 80 16\n", (HOST, PORT))
try:
    data, _ = s2.recvfrom(128)
    print(f"\n    RAM 0x80-0x8F: {data.decode().strip()}")
except Exception:
    pass
s2.close()

input("\nPress Enter to exit...")
