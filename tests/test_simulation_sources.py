"""Exercise launcher ownership with local sleep processes, never ROS or UE."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
(ROOT/'results').mkdir(exist_ok=True)
script = ROOT/'start_simulation_sources.sh'
source = script.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
namespace = {'__name__': 'launcher_test'}
exec(compile(source, str(script), 'exec'), namespace)
real_popen, real_sleep = subprocess.Popen, time.sleep
results = {}


def entry(name, pid=100):
    return dict(pid=pid, domain='10', host='192.168.10.22',
                port=namespace['SERVICES'][name][3], localhost='0')


with tempfile.TemporaryDirectory(prefix='source_launcher_', dir=ROOT/'results') as temporary:
    root = Path(temporary)/'lidar_visual_fusion'
    root.mkdir()
    runtime = root.parent/'roma_t3_algorithm_bundle_20260825/envx_runtime'
    (runtime/'install').mkdir(parents=True)
    (root/'scripts').mkdir()
    (runtime/'install/setup.bash').touch()
    for _, (_, script_name, _, _) in namespace['SERVICES'].items():
        script_path = (root/'scripts' if script_name == 'start_capture_source.sh' else runtime)/script_name
        script_path.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s %s %s" "$ROS_DOMAIN_ID" "$ROS_LOCALHOST_ONLY" "$RMW_IMPLEMENTATION" > "${0}.ready"\n'
            'exec sleep 60\n')
    found = {name: [entry(name)] for name in namespace['SERVICES']}
    original_snapshot = namespace['snapshot']
    namespace['snapshot'] = lambda: found
    with patch('subprocess.Popen', side_effect=AssertionError('Duplicate start attempted')):
        assert namespace['main'](root) == 0
    results['reuse_without_duplicate_start'] = True

    found['lunar_car_node'][0]['domain'] = '0'
    with patch('subprocess.Popen', side_effect=AssertionError('Started before conflict check')):
        try:
            namespace['main'](root)
            raise AssertionError('Domain conflict was accepted')
        except RuntimeError as error:
            assert '113658' not in str(error)  # This is a synthetic fixture.
            assert '0' in str(error)
    results['domain_conflict_blocks_all_starts'] = True

    for reuse_capture in [False, True, 'both']:
        # Each command-line invocation owns a distinct workspace/session in this fixture.
        case_root = root.parent/('case_'+str(reuse_capture))
        case_root.mkdir()
        (case_root/'scripts').mkdir()
        (case_root/'scripts/start_capture_source.sh').write_text((root/'scripts/start_capture_source.sh').read_text())
        for spec in namespace['SERVICES'].values():
            (runtime/(spec[1]+'.ready')).unlink(missing_ok=True)
        spawned = []
        sentinel = real_popen(['sleep', '60'], start_new_session=True)
        initial = {name: [] for name in namespace['SERVICES']}
        if reuse_capture:
            initial['sensor_capture_node'] = [entry('sensor_capture_node', sentinel.pid)]

        if reuse_capture == 'both':
            initial['lunar_car_node'] = [entry('lunar_car_node', sentinel.pid)]

        def launch(command, **kwargs):
            process = real_popen(command, **kwargs)
            name = next(name for name, spec in namespace['SERVICES'].items() if Path(command[1]).name == spec[1])
            spawned.append((name, process))
            return process

        def inspect_fixture():
            state = {name: list(values) for name, values in initial.items()}
            for name, process in spawned:
                if process.poll() is None:
                    state[name] = [entry(name, process.pid)]
            return state

        def interrupt_after_start(delay):
            if delay == 1:
                deadline = time.monotonic()+3
                ready_paths = [(case_root/'scripts' if name == 'sensor_capture_node' else runtime)/(namespace['SERVICES'][name][1]+'.ready') for name, _ in spawned]
                while not all(path.exists() for path in ready_paths) and time.monotonic() < deadline:
                    real_sleep(.02)
                assert all(path.read_text() == '10 0 rmw_cyclonedds_cpp' for path in ready_paths)
                raise KeyboardInterrupt
            real_sleep(delay)

        namespace['snapshot'] = inspect_fixture
        try:
            with patch('subprocess.Popen', side_effect=launch), patch('time.sleep', side_effect=interrupt_after_start):
                try:
                    namespace['main'](case_root)
                    raise AssertionError('Foreground session did not wait')
                except KeyboardInterrupt:
                    pass
            assert len(spawned) == (1 if reuse_capture == 'both' else 2 if reuse_capture else 3)
            assert all(process.poll() is not None for _, process in spawned)
            assert sentinel.poll() is None, 'Existing external process was stopped'
            results['gui_only_when_sources_exist' if reuse_capture == 'both' else 'reuse_capture_stop_owned_car_gui' if reuse_capture else 'start_three_stop_owned_groups'] = True
        finally:
            for _, process in spawned:
                if process.poll() is None:
                    os.killpg(process.pid, 15)
                    process.wait(timeout=5)
            sentinel.terminate()
            sentinel.wait(timeout=5)

namespace['snapshot'] = original_snapshot
result = dict(passed=all(results.values()), tests=results,
              qualifications='Lifecycle tests use sleep fixtures only; no connection to UE or real ROS source restarts.')
out = ROOT/'results/simulation_sources_launcher'
out.mkdir(exist_ok=True)
(out/'test_result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
