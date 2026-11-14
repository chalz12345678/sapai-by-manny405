import os
import sys

### The gui package lives at the repository root, next to sapai; make it
###   importable when pytest runs from an installed sapai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
