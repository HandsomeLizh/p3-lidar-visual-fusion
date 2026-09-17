#!/usr/bin/env python3
"""Start/stop only processes created by this workspace; never pkill ROS nodes."""
import argparse,json,os,shlex,signal,subprocess,time
import yaml
import fcntl
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"run_state.json"


def identity(pid):
    try:
        s=Path(f"/proc/{pid}/stat").read_text();f=s[s.rfind(")")+2:].split()
        return None if f[0]=="Z" else f[19]
    except OSError:return None


def alive(p):
    return bool(p and p.get("identity") and identity(p["pid"])==p["identity"])


def save(s):
    t=STATE.with_suffix(".tmp");t.write_text(json.dumps(s,indent=2));t.replace(STATE)


def spawn(s,name,args,env):
    log=Path(s["output"])/f"{name}.log"
    with log.open("ab",buffering=0) as f:
        p=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    s["processes"][name]={"pid":p.pid,"identity":identity(p.pid),"command":args,"log":str(log)}
    save(s);print(f"Started {name}: PID {p.pid}",flush=True)


def stop(s,name):
    p=s["processes"].get(name)
    if not alive(p):return
    if name=="verifier":
        (Path(s["output"])/"verifier.stop").write_text("stop requested\n")
        deadline=time.monotonic()+5
        while alive(p) and time.monotonic()<deadline:time.sleep(.1)
        if not alive(p):
            print("Stopped",name,flush=True)
            return
    # Signal only a recorded, identity-checked process group from this workspace.
    os.kill(p["pid"],signal.SIGINT) if name=="pipeline" else os.killpg(p["pid"],signal.SIGINT)
    deadline=time.monotonic()+40
    while alive(p) and time.monotonic()<deadline:time.sleep(.2)
    if alive(p):
        os.killpg(p["pid"],signal.SIGTERM)
        deadline=time.monotonic()+5
        while alive(p) and time.monotonic()<deadline:time.sleep(.2)
    if alive(p):raise RuntimeError(f"{name} has not stopped; PID retained")
    print("Stopped",name,flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("action",choices=["start","stop","status"])
    a.add_argument("--profile",default=str(ROOT/"config/simulation.yaml"))
    a.add_argument("--visual-device",choices=["cpu","cuda"])
    a.add_argument("--use-sim-time",action="store_true",help="Use external /clock, including calibrated IMU bag playback")
    a.add_argument("--bag");a.add_argument("--frames",type=int,default=0)
    a.add_argument("--rate",type=float,default=1.)
    a.add_argument("--header-time-scale",type=float,default=1.)
    a.add_argument("--blackout",default="")
    a.add_argument("--image-perturbations",default="")
    a.add_argument("--raw-image-transport",action="store_true")
    a.add_argument("--frame-index")
    a.add_argument("--rviz",action="store_true")
    a.add_argument("--verify",action="store_true",help="Start interface observer before input playback")
    a.add_argument("--verify-seconds",type=float,default=320.)
    a.add_argument("--output")
    a.add_argument("--capture",action="store_true")
    a.add_argument("--ue-host",default="192.168.10.22")
    a.add_argument("--domain",type=int)
    a.add_argument("--network",action="store_true",help="Enable DDS discovery across machines")
    x=a.parse_args()
    lock=(ROOT/"run_state.lock").open("a")
    if x.action!="status":
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("Another controller action is in progress")
    s=json.loads(STATE.read_text()) if STATE.exists() else {"processes":{}}
    if x.action=="status":
        print("ROS domain:",s.get("domain"),"Output:",s.get("output"))
        for n,p in s["processes"].items():print(n,"RUNNING" if alive(p) else "STOPPED",p["pid"],p["log"])
        return
    if x.action=="stop":
        for name in ["live_relay","monitor","capture","player","verifier","rviz","visuals","pipeline"]:stop(s,name)
        return
    if any(alive(p) for p in s["processes"].values()):
        raise RuntimeError("This workspace already has an active run; use ./status.sh or ./stop.sh")
    from source_manifest import check as check_build
    check_build()
    if x.domain is not None:os.environ["ROS_DOMAIN_ID"]=str(x.domain)
    if x.network:os.environ["ROS_LOCALHOST_ONLY"]="0"
    if x.capture and x.bag:raise ValueError("Choose live capture or bag input")
    if x.capture:
        for proc in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                command=proc.read_bytes().split(b"\0")[0].decode(errors="replace")
                if Path(command).name=="sensor_capture_node":
                    raise RuntimeError("An existing sensor_capture_node is active; select its ROS domain and omit --capture")
            except OSError:pass
    startup_profile=yaml.safe_load(Path(x.profile).read_text())
    if startup_profile.get("visual_source")=="learned":
        if x.visual_device:startup_profile["learned_visual"]["device"]=x.visual_device
        from t3_lidar_visual_fusion.learned_assets import verify_assets
        try:verify_assets(ROOT,startup_profile["learned_visual"]["backend"])
        except (OSError,RuntimeError,KeyError) as error:
            startup_profile["requested_visual_source"]="learned"
            startup_profile["visual_source"]="none"
            startup_profile["visual_disable_reason"]=str(error)[:500]
            print("Visual assets unavailable; continuing with LiDAR only: "+str(error),flush=True)
    elif x.visual_device:raise ValueError("--visual-device requires a learned profile")
    from preflight import check as check_graph
    check_graph(x.capture)
    from generate_config import generate
    out=Path(x.output) if x.output else ROOT/"results"/time.strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True,exist_ok=False)
    snapshot=out/"profile.yaml"
    snapshot.write_text(yaml.safe_dump(startup_profile,sort_keys=False))
    profile=generate(snapshot,out/"config")
    if x.bag and profile["use_imu"]:
        raise ValueError("Bundled legacy player supports stereo+LiDAR bags only; use ros2 bag play for calibrated IMU bags")
    if x.bag and profile.get("visual_source")=="learned":
        profile["normalized_camera_input"]=not x.raw_image_transport
        if profile.get("adaptive_source_selection",False):
            profile["mapping_pose_settle_sec"]=profile.get("mapping_pose_settle_sec",profile.get("visual_max_age_sec",1.5)+.1)
    if x.image_perturbations and (not x.bag or not profile.get("normalized_camera_input",False)):
        raise ValueError("Image perturbations require normalized learned-vision bag playback")
    (out/"profile.yaml").write_text(yaml.safe_dump(profile,sort_keys=False))
    s={"output":str(out),"domain":os.environ.get("ROS_DOMAIN_ID","57"),
       "localhost_only":os.environ.get("ROS_LOCALHOST_ONLY","1"),
       "cyclonedds_uri":os.environ.get("CYCLONEDDS_URI",""),"processes":{}}
    save(s)
    env=dict(os.environ)
    env["ROS_LOG_DIR"]=str(out/"ros_logs")
    spawn(s,"pipeline",["ros2","launch",str(ROOT/"scripts/fusion.launch.py"),
         "profile:="+str(snapshot.resolve()),"output_dir:="+str(out),
         "use_sim_time:="+str(bool(x.bag) or x.use_sim_time).lower()],env)
    spawn(s,"monitor",["/usr/bin/python3",str(ROOT/"scripts/watch_resources.py"),str(STATE),
          "--soft-mib",str(profile.get("process_tree_soft_mib",6144)),
          "--stop-mib",str(profile.get("process_tree_stop_mib",8192))],env)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        p=s["processes"]["pipeline"]
        if not alive(p):
            stop(s,"monitor")
            raise RuntimeError("Pipeline exited; see "+p["log"])
        with Path(p["log"]).open("rb") as stream:
            stream.seek(max(0,Path(p["log"]).stat().st_size-262144))
            text=stream.read().decode(errors="replace")
        visual_ready=(profile.get("visual_source")!="learned" or "Learned frontend ready" in text)
        if "Terrain mapper ready" in text and "Guard ready" in text and visual_ready:break
        time.sleep(.5)
    else:
        stop(s,"monitor")
        stop(s,"pipeline")
        raise RuntimeError("Pipeline did not become ready; see "+str(out/"pipeline.log"))
    if x.verify:
        ready=out/"verifier.ready"
        spawn(s,"verifier",["/usr/bin/python3",str(ROOT/"scripts/verify_run.py"),
            "--seconds",str(x.verify_seconds),"--report",str(out/"verification.json"),
            "--visual-source",profile.get("visual_source","none"),
            "--cloud-topic","/T3/mapping/stereo_map" if profile.get("mapping_source")=="stereo" else "/T3/mapping/lidar_map",
            "--ready-file",str(ready),"--stop-file",str(out/"verifier.stop")],env)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline and not ready.exists():
            if not alive(s["processes"]["verifier"]):
                raise RuntimeError("Interface observer failed before playback")
            time.sleep(.1)
        if not ready.exists():
            raise RuntimeError("Interface observer not ready before playback")
    if x.rviz:
        display_env=dict(env)
        display_env.setdefault("DISPLAY",":0")
        display_env.setdefault("XAUTHORITY",f"/run/user/{os.getuid()}/gdm/Xauthority")
        display_env["XDG_RUNTIME_DIR"]=f"/run/user/{os.getuid()}"
        display_env["T3_VISUAL_RUNTIME"]=str(out/"visual_runtime")
        (out/"visual_runtime").mkdir(exist_ok=True)
        spawn(s,"visuals",["bash",str(ROOT/"scripts/p3_visuals.sh"),"monitor",
                         "-p","use_sim_time:="+str(bool(x.bag) or x.use_sim_time).lower()],display_env)
        spawn(s,"rviz",["bash",str(ROOT/"scripts/p3_visuals.sh"),"window",
                       str(out/"visual_runtime"),"--ros-args","-p",
                       "use_sim_time:="+str(bool(x.bag) or x.use_sim_time).lower()],display_env)
        ui_deadline=time.monotonic()+15.
        while time.monotonic()<ui_deadline:
            for name in ("visuals","rviz"):
                if not alive(s["processes"][name]):
                    raise RuntimeError("Mapping remains active, but "+name+" exited; see "+s["processes"][name]["log"])
            if (out/"visual_runtime/monitor_window.json").exists():break
            time.sleep(.25)
        else:raise RuntimeError("Mapping remains active, but the map window is not ready; see "+str(out/"visuals.log"))
    if x.capture:
        spawn(s,"capture",[str(ROOT/"scripts/capture.sh"),x.ue_host,str(out)],env)
    if x.bag:
        args=["/usr/bin/python3",str(ROOT/"scripts/replay_bag.py"),x.bag,"--frames",str(x.frames),
              "--rate",str(x.rate),"--header-time-scale",str(x.header_time_scale),
              "--report",str(out/"replay.json"),"--timeline",str(out/"replay_frames.csv")]
        if x.frame_index:args+=["--frame-index",x.frame_index]
        if x.blackout:args+=["--blackout",x.blackout]
        if profile.get("normalized_camera_input",False):
            args+=["--normalized-stereo-profile",str(snapshot)]
        if x.image_perturbations:args+=["--image-perturbations",str(Path(x.image_perturbations).resolve())]
        if profile.get("visual_source")=="none":args+=["--lidar-only"]
        spawn(s,"player",args,env)
    print("STARTED; awaiting sensor/estimator validation:",out,flush=True)
    print("Outputs: /T3/semantic/current_pose, /T3/semantic/trajectory, /T3/semantic/incremental_map",flush=True)
    cloud_topic="/T3/mapping/stereo_map" if profile.get("mapping_source")=="stereo" else "/T3/mapping/lidar_map"
    print("Maps: /T3/mapping/elevation_map, "+cloud_topic,flush=True)


if __name__=="__main__":main()
