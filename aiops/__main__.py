"""
Entry point for executing the aiops package as a module:
  python3 -m aiops <evaluate|demo> [...]
"""

import sys
from aiops.cli import main

if __name__ == "__main__":
    sys.exit(main())
