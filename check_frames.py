from pathlib import Path

p = Path(r"data\mft25_mot\val_half\BT-001\gt\gt.txt")

mx = 0
for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = line.strip()
    if line:
        mx = max(mx, int(line.split(",")[0]))

print("BT-001 max_frame =", mx)