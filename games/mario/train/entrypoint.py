"""
entrypoint.py — reads config.py and launches the correct training script.
config.py is mounted as a volume so changes take effect on container restart
without rebuilding the image.
"""
import importlib.util
import os
import sys

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.py")

def load_config():
    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg

if __name__ == "__main__":
    cfg = load_config()

    print(f"[entrypoint] TRAIN_MODE={cfg.TRAIN_MODE}")
    print(f"[entrypoint] N_ENVS={cfg.N_ENVS} | TOTAL_TIMESTEPS={cfg.TOTAL_TIMESTEPS}")

    MODE = cfg.TRAIN_MODE

    if MODE == "pixel_v0":
        from train_v0 import main
        main()
    elif MODE == "pixel_v3":
        from train_v3 import main
        main()
    elif MODE in ("ram_v0", "ram_v3"):
        version = MODE.split("_")[1]
        from train_ram import main
        main(version=version)
    else:
        print(f"[entrypoint] Unknown TRAIN_MODE: {MODE}")
        sys.exit(1)
