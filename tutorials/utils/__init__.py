import sys
from pathlib import Path


tutorials_dir = Path.cwd().parent
sys.path.insert(0, str(tutorials_dir))


import etl
import plots
