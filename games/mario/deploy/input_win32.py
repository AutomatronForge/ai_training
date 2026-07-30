"""
input_win32.py — Low-level Windows key injection that works regardless of focus.
Replaces pyautogui for RetroArch key control.
"""
import ctypes
import ctypes.wintypes as wt

# Virtual key codes
VK = {
    "right":  0x27,
    "left":   0x25,
    "up":     0x26,
    "down":   0x28,
    "x":      0x58,
    "z":      0x5A,
    "enter":  0x0D,
    "shift":  0x10,
}

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         wt.WORD),
        ("wScan",       wt.WORD),
        ("dwFlags",     wt.DWORD),
        ("time",        wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wt.ULONG)),
    ]

class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("_input", INPUT_UNION)]


def _send_key(vk, key_up=False):
    flags = KEYEVENTF_KEYUP if key_up else 0
    inp = INPUT(
        type=INPUT_KEYBOARD,
        _input=INPUT_UNION(ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None))
    )
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def key_down(key: str):
    vk = VK.get(key.lower())
    if vk:
        _send_key(vk, key_up=False)


def key_up(key: str):
    vk = VK.get(key.lower())
    if vk:
        _send_key(vk, key_up=True)


def release_all():
    for vk in VK.values():
        _send_key(vk, key_up=True)
