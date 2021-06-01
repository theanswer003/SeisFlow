# -*- coding: utf-8 -*-

import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from core import main

os.makedirs('results', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

if __name__ == '__main__':
    main.run()


