"""A remote display startup must preserve an already-running vehicle map."""
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ensure_hardware104 as entry


def setup_state(tmp_path, monkeypatch, names, domain='59', source='stereo'):
    output = tmp_path / 'old_map'
    output.mkdir()
    (output / 'profile.yaml').write_text(yaml.safe_dump({'mapping_source': source, 'hardware': {'enabled': True}}))
    (tmp_path / 'run_state.json').write_text(json.dumps({'output': str(output), 'domain': domain,
        'localhost_only': '0', 'processes': {name: {'name': name} for name in names}}))
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    monkeypatch.setattr(entry, 'alive', lambda record: bool(record))
    monkeypatch.setattr(sys, 'argv', ['ensure_hardware104.py'])


@pytest.mark.parametrize('source',['stereo','range'])
def test_reuse_never_stops_or_restarts_map(tmp_path, monkeypatch,source):
    setup_state(tmp_path, monkeypatch, ['pipeline', 'monitor', 'rviz'],source=source)
    monkeypatch.setattr(entry, 'stop', lambda *_: pytest.fail('Existing map was stopped'))
    monkeypatch.setattr(entry.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('Unexpected new process'))
    attached=[]
    monkeypatch.setattr(entry,'ensure_viewer_transport',lambda state:attached.append(state['output']))
    entry.main()
    assert attached==[str(tmp_path/'old_map')]


@pytest.mark.parametrize('domain,source', [('75', 'stereo'), ('59', 'unexpected')])
def test_incompatible_active_run_is_preserved(tmp_path, monkeypatch, domain, source):
    setup_state(tmp_path, monkeypatch, ['pipeline', 'rviz'], domain=domain, source=source)
    monkeypatch.setattr(entry, 'stop', lambda *_: pytest.fail('Incompatible run was stopped'))
    monkeypatch.setattr(entry.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('Unexpected new process'))
    with pytest.raises(RuntimeError, match='未停止'):
        entry.main()


def test_only_recorded_orphan_processes_are_cleaned_before_start(tmp_path, monkeypatch):
    setup_state(tmp_path, monkeypatch, ['monitor', 'rviz'])
    stopped = []
    calls = []
    monkeypatch.setattr(entry, 'stop', lambda state, name: stopped.append(name) if name in state['processes'] else None)
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1].endswith('start_hardware.sh'):
            # The existing controller must be able to take its lock.
            with (tmp_path / 'run_state.lock').open('a') as lock:
                entry.fcntl.flock(lock, entry.fcntl.LOCK_EX | entry.fcntl.LOCK_NB)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(entry.subprocess, 'run', run)
    entry.main()
    assert stopped == ['monitor', 'rviz']
    assert [Path(command[1]).name for command, _ in calls] == ['check_hardware.py', 'start_hardware.sh']
    assert calls[-1][1]['env']['P3_HARDWARE_DOMAIN'] == '59'


def test_check_does_not_touch_stopped_run(tmp_path, monkeypatch):
    setup_state(tmp_path, monkeypatch, ['rviz'])
    monkeypatch.setattr(sys, 'argv', ['ensure_hardware104.py', '--check'])
    monkeypatch.setattr(entry, 'stop', lambda *_: pytest.fail('Read-only check stopped a process'))
    monkeypatch.setattr(entry.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('Read-only check launched a process'))
    entry.main()


def test_missing_capture_does_not_start_it_or_close_ui(tmp_path, monkeypatch):
    setup_state(tmp_path, monkeypatch, ['rviz'])
    calls = []
    monkeypatch.setattr(entry, 'stop', lambda *_: pytest.fail('Input check should run before UI cleanup'))
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout='Missing camera frames', stderr='')
    monkeypatch.setattr(entry.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='start_104_capture.sh'):
        entry.main()
    assert len(calls) == 1 and Path(calls[0][1]).name == 'check_hardware.py'
