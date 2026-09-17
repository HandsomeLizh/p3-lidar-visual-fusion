#!/usr/bin/env python3
"""Atomically install a private capture; never overwrite a running executable inode."""
import argparse
import os
from pathlib import Path
import shutil
from prepare_capture_transport import BINARY, ROOT, record


def install(candidate):
    candidate = candidate.resolve()
    if not candidate.is_relative_to(ROOT/'build') or not candidate.is_file():
        raise ValueError('Expected a built capture inside this project build directory')
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    previous = ROOT/'build/capture_transport/previous_sensor_capture_node'
    previous.parent.mkdir(parents=True, exist_ok=True)
    if BINARY.is_file():
        shutil.copy2(BINARY, previous)
    temporary = BINARY.with_name(BINARY.name+'.next')
    shutil.copy2(candidate, temporary)
    os.replace(temporary, BINARY)
    record()
    print('Installed private capture atomically. Existing processes retain their old executable.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    install(parser.parse_args().candidate)
