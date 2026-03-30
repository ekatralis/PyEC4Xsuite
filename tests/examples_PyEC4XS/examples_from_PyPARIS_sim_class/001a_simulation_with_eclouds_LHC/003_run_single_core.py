import os
import sys
import importlib
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(SCRIPT_DIR)

sim_mod = importlib.import_module("PyEC4XS.Simulation")

ring = sim_mod.get_serial_CPUring(param_file=str(SCRIPT_DIR / "Simulation_parameters.py"))
ring.run()
