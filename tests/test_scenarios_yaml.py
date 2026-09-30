"""The scripted demo scenarios (scenarios/*.yaml) must keep passing their own `expect` blocks."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("scenario_runner", ROOT / "scenarios" / "run.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def test_there_is_a_yaml_for_each_demo_use_case():
    assert {"incar", "support_mid", "support_after", "field", "access"} <= set(runner.all_names())


@pytest.mark.parametrize("name", runner.ORDER)
async def test_scenario_meets_its_expectations(name, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)  # traces go to tmp, not the repo
    problems = await runner.play(name, "mock", html=False, verbose=False)
    assert problems == []
