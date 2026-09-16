#!/usr/bin/env python3
"""Capture only the RViz window owned by the selected test output."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from control import alive

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--name',default='rviz_complete.png');args=parser.parse_args()
    output=Path(args.output).resolve();target=output/args.name
    if not output.is_relative_to(ROOT/'results') or target.parent!=output:
        raise ValueError('Screenshot must be inside the selected result directory')
    state=json.loads((ROOT/'run_state.json').read_text())
    owned=state.get('processes',{}).get('rviz')
    if state.get('output')!=str(output) or not owned or not alive(owned):
        raise RuntimeError('No live RViz owned by this output')
    os.environ.update(DISPLAY=':0',XAUTHORITY='/run/user/1000/gdm/Xauthority',XDG_RUNTIME_DIR='/run/user/1000')
    text=subprocess.run(['xwininfo','-root','-tree'],capture_output=True,text=True,check=True).stdout
    candidates=[]
    for line in text.splitlines():
        match=re.search(r'^\s*(0x[0-9a-fA-F]+).*?\s(\d+)x(\d+)[+-]',line)
        if not match:continue
        window,width,height=match.groups();width,height=int(width),int(height)
        if width<600 or height<400:continue
        property_text=subprocess.run(['xprop','-id',window,'_NET_WM_PID'],capture_output=True,text=True).stdout
        pid=re.search(r'=\s*(\d+)',property_text)
        if pid and int(pid.group(1))==owned['pid']:candidates.append((width*height,int(window,16)))
    if not candidates:raise RuntimeError('Owned RViz window is not visible')
    from PyQt5.QtWidgets import QApplication
    app=QApplication([]);window=max(candidates)[1]
    image=app.primaryScreen().grabWindow(window)
    if image.isNull() or not image.save(str(target)):raise RuntimeError('Screenshot could not be saved')
    print(str(target),flush=True)


if __name__=='__main__':main()
