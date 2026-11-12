"""Makes the repo root importable from tests/ (the project has no src/
package layout -- config.py, dqn.py, etc. are top-level modules)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
