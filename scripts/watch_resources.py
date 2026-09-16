#!/usr/bin/env python3
"""Watch only recorded process identities. Rotate logs; stop owned run on exhaustion."""
import argparse,json,os,shutil,time,signal
from pathlib import Path
from control import alive,stop,identity


def descendants(roots):
    parent={};rss={}
    for path in Path("/proc").glob("[0-9]*"):
        try:
            s=(path/"stat").read_text();f=s[s.rfind(")")+2:].split()
            pid=int(path.name);parent[pid]=int(f[1])
            pages=int((path/"statm").read_text().split()[1]);rss[pid]=pages*os.sysconf("SC_PAGE_SIZE")/2**20
        except (OSError,ValueError,IndexError):
            continue
    selected=set(roots)
    while True:
        children={pid for pid,ppid in parent.items() if ppid in selected}
        new=children-selected
        if not new:break
        selected.update(new)
    return {pid:rss.get(pid,0.) for pid in selected}


def rotate(path,limit=16*2**20):
    if not path.exists() or path.stat().st_size<=limit:return
    # Log files are disposable diagnostics; maps and inputs are never removed.
    for number in [2,1]:
        source=path.with_name(path.name+"."+str(number))
        if source.exists():source.replace(path.with_name(path.name+"."+str(number+1)))
    shutil.copyfile(path,path.with_name(path.name+".1"))
    with path.open("r+b") as stream:stream.truncate(0)



def shed_visual_processes(pids,root):
    stopped=[]
    for pid in pids:
        try:
            start=identity(pid)
            args=Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            paths=[Path(a.decode()) for a in args[:3] if a]
            visual=any(p.name in ("learned_odometry","vins_node") and p.is_absolute()
                       and p.is_relative_to(root) for p in paths)
            if visual and start and identity(pid)==start:
                os.kill(pid,signal.SIGTERM);stopped.append(pid)
        except (OSError,UnicodeError):pass
    return stopped


def main():
    a=argparse.ArgumentParser();a.add_argument("state");a.add_argument("--soft-mib",type=float,default=6144)
    a.add_argument("--stop-mib",type=float,default=8192);x=a.parse_args()
    state=json.loads(Path(x.state).read_text());out=Path(state["output"])
    strikes=0
    visual_shed=False
    peak_total=0.
    root=Path(__file__).resolve().parents[1]
    while alive(state["processes"].get("pipeline")):
        state=json.loads(Path(x.state).read_text())
        roots=[p["pid"] for name,p in state["processes"].items() if name!="monitor" and alive(p)]
        values=descendants(roots);total=sum(values.values())
        peak_total=max(peak_total,total)
        available=next(int(l.split()[1])/1024. for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
        disk=shutil.disk_usage(out).free/2**30
        record=dict(wall_time=time.time(),tree_rss_mib=total,tree_rss_peak_mib=peak_total,system_available_mib=available,
            disk_free_gib=disk,soft_limit_mib=x.soft_mib,stop_limit_mib=x.stop_mib,
            memory_warning=total>x.soft_mib,process_rss_mib=values)
        with (out/"resources.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
        tmp=out/"resources_latest.json.tmp";tmp.write_text(json.dumps(record,indent=2))
        tmp.replace(out/"resources_latest.json")
        for path in list(out.glob("*.log"))+list(out.glob("*.jsonl")):
            rotate(path)
        for path in (out/"ros_logs").rglob("*.log"):rotate(path)
        if not visual_shed and (total>x.soft_mib or available<1024):
            stopped=shed_visual_processes(values,root)
            if stopped:
                visual_shed=True
                (out/"visual_unavailable.json").write_text(json.dumps(dict(
                    reason="resource_pressure",action="continue_lidar_only",stopped_visual_pids=stopped)))
                print("Disabled only owned visual frontend under resource pressure",stopped,flush=True)
        severe=total>x.stop_mib or available<512 or disk<1.
        strikes=strikes+1 if severe else 0
        if strikes>=3:
            (out/"resource_stop.json").write_text(json.dumps(record,indent=2))
            for name in ["capture","player","rviz","visuals","pipeline"]:
                try:stop(state,name)
                except Exception as error:print("Resource stop failed for",name,str(error),flush=True)
            return
        time.sleep(5.)


if __name__=="__main__":main()
