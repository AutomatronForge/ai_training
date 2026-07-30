"""
test_ram_scan.py — Scan RAM to find where Mario's x actually changes
Move Mario manually in RetroArch while this runs, and it will find
which address changes.
"""
import socket
import time

HOST = "127.0.0.1"
PORT = 55355


def read_range(sock, start, count):
    cmd = f"READ_CORE_RAM {start:X} {count}\n".encode()
    sock.sendto(cmd, (HOST, PORT))
    try:
        data, _ = sock.recvfrom(512)
        parts = data.decode().strip().split()
        return [int(p, 16) for p in parts[2:2+count]]
    except Exception:
        return [0] * count


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(0.5)

print("RAM Scanner — finds which addresses change when Mario moves")
print("=" * 55)
print()
print("Step 1: Make sure Mario is standing still in RetroArch")
input("Press Enter when Mario is NOT moving...")

# Take baseline snapshot of first 512 bytes
print("Taking baseline snapshot...")
baseline = []
for i in range(0, 512, 32):
    baseline.extend(read_range(s, i, 32))
print(f"Snapshot taken ({len(baseline)} bytes)")

print()
print("Step 2: NOW move Mario RIGHT in RetroArch for 2-3 seconds")
print("        (press and hold right arrow key in RetroArch window)")
input("Press Enter when Mario has moved right...")

# Take second snapshot
current = []
for i in range(0, 512, 32):
    current.extend(read_range(s, i, 32))

# Find changed addresses
print()
print("Addresses that changed:")
changed = []
for i, (b, c) in enumerate(zip(baseline, current)):
    if b != c:
        changed.append((i, b, c, c - b))
        print(f"  0x{i:04X} : {b:3d} -> {c:3d}  (delta={c-b:+d})")

if not changed:
    print("  No addresses changed — Mario did not move or ROM not loaded correctly")
    print()
    print("  Try: Load the ROM again in RetroArch")
    print("       Make sure you're using the NES Super Mario Bros (USA) ROM")
    print("       Core: Mesen or FCEUmm")
else:
    print()
    # Find most likely x address (positive delta, reasonable range)
    candidates = [(a, b, c, d) for a, b, c, d in changed if 0 < d < 50]
    if candidates:
        print("Most likely x position addresses (small positive delta):")
        for a, b, c, d in candidates[:5]:
            print(f"  0x{a:04X} = {c} (was {b}, moved +{d})")

s.close()
input("\nPress Enter to exit...")
