"""
test_find_x.py — Verify x position address by watching it change in real time
"""
import socket
import time

HOST = "127.0.0.1"
PORT = 55355

CANDIDATES = [0x0005, 0x0006, 0x0007, 0x0008, 0x006D, 0x006E, 0x006F, 0x0070]

def read_ram(sock, addr):
    cmd = f"READ_CORE_RAM {addr:X} 1\n".encode()
    sock.sendto(cmd, (HOST, PORT))
    try:
        data, _ = sock.recvfrom(64)
        parts = data.decode().strip().split()
        return int(parts[2], 16) if len(parts) >= 3 else 0
    except:
        return 0

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(0.5)

print("Move Mario right in RetroArch. Watch which value increases:")
print(f"{'Address':<12} {'Value':>6}")
print("-" * 20)

for _ in range(20):
    vals = {addr: read_ram(s, addr) for addr in CANDIDATES}
    line = " | ".join(f"0x{a:04X}={v:3d}" for a, v in vals.items())
    print(f"\r{line}", end="", flush=True)
    time.sleep(0.2)

print("\n\nDone.")
s.close()
input("Press Enter to exit...")
