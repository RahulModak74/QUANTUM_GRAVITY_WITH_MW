# Shim: lets the unchanged March-2026 THEORY scripts load the Nov-2026 v5 model
# (the architecture is identical; only the data/prior code changed in v5).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'code'))
from qg_toy_vae_trainer_v5 import QGPhysicsVAE  # noqa: F401
