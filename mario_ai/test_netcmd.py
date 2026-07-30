"""
test_netcmd.py — Send joypad input to RetroArch via network commands instead of pyautogui
This bypasses the keyboard focus issue entirely.
"""
import socket
import time

HOST = "127.0.0.1"
PORT = 55355

# RetroArch joypad button IDs (RetroPad layout)
RETRO_DEVICE_ID_JOYPAD_RIGHT = 4
RETRO_DEVICE_ID_JOYPAD_LEFT  = 3
RETRO_DEVICE_ID_JOYPAD_A     = 8   # Jump (NES A)
RETRO_DEVICE_ID_JOYPAD_B     = 0   # Run  (NES B)
RETRO_DEVICE_ID_JOYPAD_START = 6


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
    low    = read_ram(sock, 0x006D)
    screen = read_ram(sock, 0x006E)
    return screen * 256 + low


def send_cmd(sock, cmd):
    sock.sendto(cmd.encode(), (HOST, PORT))
    try:
        data, _ = sock.recvfrom(64)
        return data.decode().strip()
    except Exception:
        return ""


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(0.5)

print("=" * 50)
print("RetroArch Network Joypad Test")
print("=" * 50)

# Check connection
resp = send_cmd(s, "VERSION")
print(f"RetroArch: {resp}")

print(f"\nInitial x position: {get_x(s)}")
print("\nTesting network joypad commands...")

# Try SET_CONTROLLER_INFO
print("\n[A] Trying JOYPAD_PRESSED D-Pad Right...")
resp = send_cmd(s, "JOYPAD_PRESSED 0 4 1")  # port 0, id 4 (right), value 1
print(f"    Response: '{resp}'")

# Try pressing right via network cmd
print("\n[B] Trying SET_CONTROLLER_INFO...")
for cmd in [
    "JOYPAD_PRESSED 0 4 1",
    "JOYPAD_PRESSED 0 4",
    "JOYPAD_PRESSED 4",
    "INPUT_PRESS 0 right",
    "INPUT right 1",
]:
    resp = send_cmd(s, cmd)
    x = get_x(s)
    print(f"    CMD: '{cmd}' -> resp='{resp}' x={x}")
    time.sleep(0.2)

print(f"\nFinal x position: {get_x(s)}")

print("\n" + "=" * 50)
print("If none of the above moved Mario, we need to use")
print("a different approach: win32api direct key injection")
print("instead of pyautogui.")
print("=" * 50)

# Try win32api approach
print("\n[C] Trying win32api direct key injection...")
try:
    import ctypes
    import win32api
    import win32con
    import win32gui

    # Find RetroArch window handle
    hwnd = win32gui.FindWindow(None, "RetroArch Mesen 0.9.9")
    if not hwnd:
        # Try partial match
        def enum_cb(h, results):
            title = win32gui.GetWindowText(h)
            if "retroarch" in title.lower():
                results.append((h, title))
        results = []
        win32gui.EnumWindows(enum_cb, results)
        if results:
            hwnd, title = results[0]
            print(f"    Found window: '{title}' (hwnd={hwnd})")
        else:
            hwnd = None

    if hwnd:
        print(f"    Sending VK_RIGHT directly to hwnd={hwnd}")
        x_before = get_x(s)

        # Send key directly to window regardless of focus
        VK_RIGHT = 0x27
        win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, VK_RIGHT, 0)
        time.sleep(0.5)
        win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, VK_RIGHT, 0)
        time.sleep(0.5)
        win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, VK_RIGHT, 0)
        time.sleep(0.5)
        win32api.PostMessage(hwnd, win32con.WM_KEYUP, VK_RIGHT, 0)

        x_after = get_x(s)
        print(f"    x before: {x_before}, x after: {x_after}")
        if x_after > x_before:
            print("    SUCCESS! win32api works!")
        else:
            print("    FAIL — win32api also not working")
    else:
        print("    Could not find RetroArch window handle")

except ImportError:
    print("    win32api not installed — installing...")
    import subprocess
    subprocess.run(["pip", "install", "pywin32"], check=True)
    print("    Installed. Run this script again.")
except Exception as e:
    print(f"    ERROR: {e}")

s.close()
input("\nPress Enter to exit...")
