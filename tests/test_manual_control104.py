"""Controller entry must reuse the chassis and validate the display before enabling it."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import manual_control104 as entry


def test_existing_chassis_is_not_reinitialized(monkeypatch):
    monkeypatch.setattr(entry, 'running', lambda _: [{'pid': 10, 'domain': '19', 'args': [str(entry.DRIVER)]}])
    monkeypatch.setattr(entry.subprocess, 'run', lambda *_a, **_k: pytest.fail('Existing chassis must be reused'))
    entry.ensure_driver()


@pytest.mark.parametrize('domain,path', [('59', str(entry.DRIVER)), ('19', '/different/chassis_node')])
def test_conflicting_chassis_is_left_alone(monkeypatch, domain, path):
    monkeypatch.setattr(entry, 'running', lambda _: [{'pid': 10, 'domain': domain, 'args': [path]}])
    monkeypatch.setattr(entry.subprocess, 'run', lambda *_a, **_k: pytest.fail('Conflicting chassis must be left alone'))
    with pytest.raises(RuntimeError, match='未停止'):
        entry.ensure_driver()


def fixture_paths(tmp_path, monkeypatch):
    driver = tmp_path / 'driver'
    gui = tmp_path / 'gui.py'
    for name in ('driver', 'gui.py', 'start_mars_car.sh'):
        (tmp_path / name).touch()
    for name, value in [('ROOT', tmp_path), ('CHASSIS', tmp_path), ('DRIVER', driver), ('GUI', gui)]:
        monkeypatch.setattr(entry, name, value)
    monkeypatch.setattr(entry, 'running', lambda _: [])
    monkeypatch.setattr(sys, 'argv', ['manual_control104.py'])


def test_display_failure_cannot_start_chassis(tmp_path, monkeypatch):
    fixture_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(entry, 'check_display', lambda: (_ for _ in ()).throw(RuntimeError('no display')))
    monkeypatch.setattr(entry, 'ensure_driver', lambda: pytest.fail('Display validation must precede motor startup'))
    with pytest.raises(RuntimeError, match='no display'):
        entry.main()


def test_check_does_not_open_gui_or_start_driver(tmp_path, monkeypatch):
    fixture_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['manual_control104.py', '--check'])
    monkeypatch.setattr(entry, 'check_display', lambda: pytest.fail('Check opened a window'))
    monkeypatch.setattr(entry, 'ensure_driver', lambda: pytest.fail('Check started a driver'))
    entry.main()
