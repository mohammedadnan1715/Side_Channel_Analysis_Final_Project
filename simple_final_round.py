#!/usr/bin/env python3
"""Run the simple final-round method; shared implementation is alongside this file."""
import sys
from aes_timing_attack import main

if __name__ == '__main__':
    raise SystemExit(main(['--method', 'simple', *sys.argv[1:]]))
