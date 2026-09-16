#!/usr/bin/env python3
"""Build a private copy of P3 capture with overlapping receive/decode stages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'build/capture_transport/source.json'
BINARY=ROOT/'install/capture_transport/lib/sensor_capture_ros2/sensor_capture_node'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(runtime):
    source=runtime/'src/sensor_capture_ros2'
    target=ROOT/'vendor/capture_overlap_source'
    if not (source/'CMakeLists.txt').is_file():raise FileNotFoundError(source)
    if not target.resolve().is_relative_to(ROOT/'vendor'):raise ValueError('Private source path escaped workspace')
    original={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    shutil.copytree(source,target,dirs_exist_ok=True)
    def replace(file,old,new):
        p=target/file;s=p.read_text()
        if s.count(old)!=1:raise RuntimeError(f'Capture source changed at {file}; refusing an ambiguous patch')
        p.write_text(s.replace(old,new))
    header='include/sensor_capture_ros2/sensor_capture_node.hpp'
    cpp='src/sensor_capture_node.cpp'
    replace(header,'        std::vector<uint8_t> file_data;',
            '        std::vector<uint8_t> file_data;\n        cv::Mat prepared_image; // decoded while the network receiver is active')
    replace(header,'\n                           const std::string & file_name);',
            '\n                           const std::string & file_name, const cv::Mat & prepared = cv::Mat());')
    replace(header,'\n                      const std::vector<uint8_t> & data, const rclcpp::Time & stamp);',
            '\n                      const std::vector<uint8_t> & data, const rclcpp::Time & stamp, const cv::Mat & prepared = cv::Mat());')
    replace(cpp,'                batch_items_.push_back(std::move(item));','''                const std::string key = makePublisherKey(stype, item.header.camera_id);
                const bool selected = publishers_.count(key) || tof_depth_publishers_.count(key);
                if (selected) {
                    if (stype == SENSOR_RGB && publishers_.count(key)) {
                        // Decode only selected images while receiveThread reads
                        // the remaining files. Publication still waits for the
                        // complete batch and its validated common meta stamp.
                        item.prepared_image = cv::imdecode(cv::Mat(item.file_data, false), cv::IMREAD_UNCHANGED);
                        if (!item.prepared_image.empty()) std::vector<uint8_t>().swap(item.file_data);
                    }
                    batch_items_.push_back(std::move(item));
                } // Unselected payloads are released immediately after optional saving.''')
    replace(cpp,'publishSensorData(item.header, item.file_data, item.header.file_name);',
            'publishSensorData(item.header, item.file_data, item.header.file_name, item.prepared_image);')
    replace(cpp,'    const std::string & /*file_name*/)','    const std::string & /*file_name*/, const cv::Mat & prepared)')
    replace(cpp,'            publishImage(base_key, file_data, stamp);','            publishImage(base_key, file_data, stamp, prepared);')
    replace(cpp,'            publishTOFPointCloud(base_key, file_data, cfg, stamp);',
            '            if (pub_it != publishers_.end()) publishTOFPointCloud(base_key, file_data, cfg, stamp);')
    signature='void SensorCaptureNode::publishImage(\n    const std::string & key,\n    const std::vector<uint8_t> & data, const rclcpp::Time & stamp)'
    replace(cpp,signature,signature[:-1]+', const cv::Mat & prepared)')
    replace(cpp,'    cv::Mat decoded = cv::imdecode(\n        cv::Mat(data, false), cv::IMREAD_UNCHANGED);',
            '    cv::Mat decoded = prepared.empty() ? cv::imdecode(\n        cv::Mat(data, false), cv::IMREAD_UNCHANGED) : prepared;')
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    MANIFEST.write_text(json.dumps({'original_source':str(source),'original_hashes':original},indent=2))

def record():
    data=json.loads(MANIFEST.read_text());data['binary_sha256']=sha(BINARY)
    MANIFEST.write_text(json.dumps(data,indent=2))

def check():
    if json.loads(MANIFEST.read_text()).get('binary_sha256')!=sha(BINARY):
        raise RuntimeError('Private capture differs from its build; rerun scripts/build_capture_transport.sh')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path)
    parser.add_argument('--record',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:check()
    elif args.record:record()
    elif args.runtime:prepare(args.runtime.resolve())
    else:parser.error('Choose --runtime, --record or --check')
