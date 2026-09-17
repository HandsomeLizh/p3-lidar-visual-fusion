"""Display startup must confirm readiness without restarting the live mapper."""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import control


def setup_state(tmp_path,monkeypatch):
    monkeypatch.setattr(control,'alive',lambda p:bool(p and p.get('running')))
    monkeypatch.setattr(control,'stop',lambda *_:pytest.fail('Mapping must remain running'))
    return dict(domain='59',localhost_only='0',cyclonedds_uri='file:///mapping.xml',
                output=str(tmp_path),processes={'pipeline':dict(pid=42,running=True)})


def ready(tmp_path,pid):
    (tmp_path/'viewer_transport_status.json').write_text(json.dumps(
        dict(pid=pid,ready=True,updated_at=time.time())))


def test_reuse_never_spawns_again(tmp_path,monkeypatch):
    state=setup_state(tmp_path,monkeypatch)
    state['processes']['viewer_transport']=dict(pid=43,running=True,log='viewer.log')
    ready(tmp_path,43)
    monkeypatch.setattr(control,'spawn',lambda *_:pytest.fail('Duplicate sender started'))
    control.ensure_viewer_transport(state)


def test_new_sender_uses_recorded_mapping_domain_and_dds(tmp_path,monkeypatch):
    state=setup_state(tmp_path,monkeypatch)
    def spawn(s,name,args,env):
        assert name=='viewer_transport' and s['processes']['pipeline']['pid']==42
        assert env['ROS_DOMAIN_ID']=='59' and env['ROS_LOCALHOST_ONLY']=='0'
        assert env['CYCLONEDDS_URI']=='file:///mapping.xml'
        s['processes'][name]=dict(pid=43,running=True,log='viewer.log')
        ready(tmp_path,43)
    monkeypatch.setattr(control,'spawn',spawn)
    control.ensure_viewer_transport(state)


def test_dead_sender_is_reported_even_with_old_ready_file(tmp_path,monkeypatch):
    state=setup_state(tmp_path,monkeypatch);ready(tmp_path,43)
    def spawn(s,name,args,env):
        s['processes'][name]=dict(pid=44,running=False,log='viewer.log')
    monkeypatch.setattr(control,'spawn',spawn)
    with pytest.raises(RuntimeError,match='Mapping remains active'):
        control.ensure_viewer_transport(state)
    assert state['processes']['pipeline']['running']


def test_display_cannot_start_without_mapper(tmp_path,monkeypatch):
    state=setup_state(tmp_path,monkeypatch);state['processes']['pipeline']['running']=False
    monkeypatch.setattr(control,'spawn',lambda *_:pytest.fail('Sender started without mapping'))
    with pytest.raises(RuntimeError,match='Mapping must be running'):
        control.ensure_viewer_transport(state)
