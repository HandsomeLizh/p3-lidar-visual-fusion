#!/usr/bin/env python3
"""Replay only measured stereo/LiDAR messages with explicit header-time correction.

No odometry, TF or reference pose is ever sent to the estimators.
"""
import argparse
import csv
import copy
import json
from pathlib import Path
import sqlite3
import time
import yaml
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image,PointCloud2
from rosgraph_msgs.msg import Clock

TOPICS=["/Car/T5/Cam_Left/image_raw/color","/Car/T5/Cam_Right/image_raw/color","/Car/T5/OS1/points"]


def batches(directory,limit,lidar_only=False):
    directory=Path(directory)
    meta=yaml.safe_load((directory/"metadata.yaml").read_text())["rosbag2_bagfile_information"]
    files=meta["relative_file_paths"]
    pending={};count=0
    for filename in files:
        db=sqlite3.connect("file:"+str(directory/filename)+"?mode=ro",uri=True)
        wanted=TOPICS[2:] if lidar_only else TOPICS
        table={i:n for i,n in db.execute("SELECT id,name FROM topics") if n in wanted}
        if not table:db.close();continue
        query="SELECT topic_id,data FROM messages WHERE topic_id IN ("+",".join("?" for _ in table)+") ORDER BY timestamp"
        for topic_id,data in db.execute(query,tuple(table)):
            name=table[topic_id];kind=PointCloud2 if name==TOPICS[2] else Image
            m=deserialize_message(data,kind)
            stamp=m.header.stamp.sec*1000000000+m.header.stamp.nanosec
            group=pending.setdefault(stamp,{})
            group[name]=m
            if len(group)==len(wanted):
                yield stamp,group
                del pending[stamp];count+=1
                if limit>0 and count>=limit:db.close();return
            while pending and (len(pending)>2 or
                sum(len(m.data) for group in pending.values() for m in group.values())>96*2**20):
                del pending[min(pending)]
        db.close()


def main():
    a=argparse.ArgumentParser()
    a.add_argument("bag");a.add_argument("--frames",type=int,default=0)
    a.add_argument("--rate",type=float,default=1.)
    a.add_argument("--header-time-scale",type=float,default=1.)
    a.add_argument("--blackout",default="",help="Frame interval, e.g. 30:40; applies to both cameras")
    a.add_argument("--report",default="")
    a.add_argument("--timeline",default="",help="Streaming CSV of actual input publication times")
    a.add_argument("--lidar-only",action="store_true")
    a.add_argument("--images-only",action="store_true")
    a.add_argument("--normalized-stereo-profile",default="",help="Publish identical adapter-normalized pixels directly on /fusion/left,right")
    a.add_argument("--frame-index",default="")
    a.add_argument("--image-perturbations",default="")
    args=a.parse_args()
    if args.lidar_only and args.images_only:raise ValueError("Choose only one input mode")
    if args.rate<=0 or args.header_time_scale<=0:raise ValueError("Rate and scale must be positive")
    if args.frame_index and args.header_time_scale!=1.:raise ValueError("Indexed receive-time playback requires scale=1")
    blackout=tuple(map(int,args.blackout.split(":"))) if args.blackout else (-1,-1)
    normalizer=None
    if args.normalized_stereo_profile:
        if args.lidar_only:raise ValueError("Normalized stereo requires camera input")
        from stereo_transport import StereoNormalizer
        normalizer=StereoNormalizer(yaml.safe_load(Path(args.normalized_stereo_profile).read_text()))
    perturbations=None
    if args.image_perturbations:
        if normalizer is None:raise ValueError("Photometric perturbations require normalized stereo")
        from image_perturbations import ImagePerturbations
        perturbations=ImagePerturbations(json.loads(Path(args.image_perturbations).read_text()))
    rclpy.init();n=Node("fusion_bag_player")
    wanted=TOPICS[:2] if args.images_only else (TOPICS[2:] if args.lidar_only else TOPICS)
    destination={t:("/fusion/left" if t==TOPICS[0] else "/fusion/right") if normalizer and t!=TOPICS[2] else t for t in wanted}
    pubs={topic:n.create_publisher(PointCloud2 if topic==TOPICS[2] else Image,destination[topic],3) for topic in wanted}
    clock=n.create_publisher(Clock,"/clock",10)
    time.sleep(2.)
    start_wall=time.monotonic()+1.;base=1000000000000;first=None;count=0
    timeline=Path(args.timeline).open("w",newline="") if args.timeline else None
    writer=csv.writer(timeline) if timeline else None
    if writer:writer.writerow(["frame","raw_header_ns","sensor_stamp_sec","publish_start_monotonic_sec","publish_end_monotonic_sec"])
    def tick():
        ns=base+max(0,int((time.monotonic()-start_wall)*args.rate*1e9))
        m=Clock();m.clock.sec=ns//1000000000;m.clock.nanosec=ns%1000000000
        clock.publish(m);rclpy.spin_once(n,timeout_sec=0)
    if args.frame_index:
        from indexed_bag import groups
        source=groups(args.bag,args.frame_index,args.frames,args.lidar_only)
    else:source=batches(args.bag,args.frames,args.lidar_only)
    for raw,group in source:
        if first is None:first=raw
        corrected=base+int(round((raw-first)*args.header_time_scale))
        due=start_wall+(corrected-base)/1e9/args.rate
        while time.monotonic()<due:
            tick();time.sleep(.01)
        tick()
        publish_start=time.monotonic()
        for topic in (TOPICS[:2] if args.images_only else ([TOPICS[2]] if args.lidar_only else [TOPICS[2],TOPICS[0],TOPICS[1]])):
            msg=normalizer.convert(group[topic],TOPICS.index(topic)) if normalizer and topic!=TOPICS[2] else group[topic]
            msg.header.stamp.sec=corrected//1000000000;msg.header.stamp.nanosec=corrected%1000000000
            if topic!=TOPICS[2] and blackout[0]<=count<blackout[1]:
                msg.data=bytes(len(msg.data))
            if perturbations and topic!=TOPICS[2]:msg=perturbations.apply(msg,count)
            pubs[topic].publish(msg)
        if writer:
            writer.writerow([count,raw,f"{corrected/1e9:.9f}",publish_start,time.monotonic()])
            timeline.flush()
        count+=1
        print(json.dumps({"frame":count,"sensor_time":corrected/1e9,"black":blackout[0]<=count-1<blackout[1],"perturbation":perturbations.kind(count-1) if perturbations else "clean"}),flush=True)
        rclpy.spin_once(n,timeout_sec=0)
    deadline=time.monotonic()+3.
    while time.monotonic()<deadline:tick();time.sleep(.02)
    if timeline:timeline.close()
    if args.report:
        Path(args.report).write_text(json.dumps({ "perturbations":perturbations.schedule if perturbations else [],"input_topics":list(destination.values()),"normalization_profile":args.normalized_stereo_profile,"frames":count,"bag":args.bag,"rate":args.rate,"header_time_scale":args.header_time_scale,"blackout":blackout,"first_raw_header_ns":first,"replay_base_ns":base,"replay_wall_elapsed_sec":time.monotonic()-start_wall,"timeline":args.timeline,"frame_index":args.frame_index},indent=2))
    n.destroy_node();rclpy.shutdown()


if __name__=="__main__":main()
