"""Publication layout and portable command interfaces; no model downloads."""
import argparse
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL = shutil.which("bash")


def test_no_site_paths_or_binary_artifacts():
    forbidden = (r"/(?:mdr\d+|home)/", r"env_[a-z]+_fedlora\.sh")
    binary_suffixes = {".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".pyc"}
    found = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix in binary_suffixes:
            found.append(str(path.relative_to(ROOT)))
        elif path.suffix in {".py", ".sh", ".yaml", ".md", ".txt"}:
            content = path.read_text(encoding="utf-8-sig")
            if any(re.search(pattern, content) for pattern in forbidden):
                found.append(str(path.relative_to(ROOT)))
    assert not found, found


def test_retained_configs_are_at_repository_root():
    for name in ("barylora_glue.yaml", "barylora_gsm8k_llama8b.yaml",
                 "barylora_gsm8k_qwen9b.yaml"):
        assert (ROOT / "configs" / name).is_file()


def test_release_has_no_explanatory_documents_or_old_installer():
    assert not (ROOT / "docs").exists()
    assert not (ROOT / "setup.py").exists()
    assert not list(ROOT.rglob("*.md"))
    assert not list(ROOT.rglob("*.json"))
    assert all(path.name.startswith("requirements") for path in ROOT.rglob("*.txt"))


@pytest.fixture
def capture_python(tmp_path):
    executable = tmp_path / "fake python"
    executable.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['CAPTURE'], 'w') as handle:\n"
        "    json.dump({'argv':sys.argv[1:], 'cwd':os.getcwd(), "
        "'path':os.environ.get('PYTHONPATH'), "
        "'offline':os.environ.get('HF_HUB_OFFLINE')}, handle)\n",
        encoding="utf-8")
    executable.chmod(0o755)
    # Keep only process-launch essentials: pytest can display fixture values
    # on failure, so credentials must never enter the captured environment.
    env = {key: os.environ[key] for key in ("PATH", "HOME", "SYSTEMROOT", "TMPDIR")
           if key in os.environ}
    env.update(ENV_PY=str(executable), CAPTURE=str(tmp_path / "args.json"),
               HF_HOME=str(tmp_path / "hf cache"), OUT=str(tmp_path / "outputs"),
               OFFLINE="1")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "HF_EVALUATE_OFFLINE", "DATA_ROOT", "MODEL_CACHE"):
        env.pop(key, None)
    return env


def launch(script, args, env, tmp_path):
    assert (ROOT / script).is_file(), script
    result = subprocess.run([SHELL, str(ROOT / script), *args], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return json.loads(Path(env["CAPTURE"]).read_text())


@pytest.mark.skipif(os.name != "posix" or not SHELL, reason="portable Bash interface runs on POSIX")
def test_glue_launcher_is_portable_and_preserves_defaults(capture_python, tmp_path):
    captured = launch("scripts/run_glue.sh", ["rte", "3", "42", "250", "0"],
                      capture_python, tmp_path)
    args = captured["argv"]
    assert captured["cwd"] == str(ROOT)
    assert captured["path"].split(os.pathsep)[0] == str(ROOT)
    assert captured["offline"] == "1"
    assert args[args.index("data.root") + 1] == str(tmp_path / "hf cache/datasets")
    assert args[args.index("llm.cache.model") + 1] == str(tmp_path / "hf cache/hub")
    assert args[args.index("federate.total_round_num") + 1] == "250"
    assert "train.local_update_steps" not in args
    assert "dataloader.batch_size" not in args


@pytest.mark.skipif(os.name != "posix" or not SHELL, reason="portable Bash interface runs on POSIX")
def test_glue_launcher_propagates_explicit_smoke_overrides(capture_python, tmp_path):
    overrides = ["dataloader.batch_size", "16", "train.local_update_steps", "2"]
    captured = launch("scripts/run_glue.sh", ["rte", "3", "42", "3", "0", *overrides],
                      capture_python, tmp_path)
    assert captured["argv"][-4:] == overrides


@pytest.mark.skipif(os.name != "posix" or not SHELL, reason="portable Bash interface runs on POSIX")
def test_llm_launcher_requires_a_model_and_preserves_config(capture_python, tmp_path):
    capture_python["MODEL_PATH"] = str(tmp_path / "local model")
    captured = launch("scripts/run_llm.sh", ["llama8b", "3", "42", "100", "0"],
                      capture_python, tmp_path)
    args = captured["argv"]
    assert "configs/barylora_gsm8k_llama8b.yaml" in args
    assert args[args.index("model.type") + 1] == capture_python["MODEL_PATH"] + "@huggingface_llm"
    assert "eval.llm_generation" not in args
    capture_python.pop("MODEL_PATH")
    result = subprocess.run([SHELL, str(ROOT / "scripts/run_llm.sh")],
                            cwd=tmp_path, env=capture_python, text=True, capture_output=True)
    assert result.returncode != 0 and "MODEL_PATH" in result.stderr


def test_inference_defaults_follow_user_cache_and_training_token_length(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    monkeypatch.delenv("MODEL_CACHE", raising=False)
    monkeypatch.delenv("DATA_ROOT", raising=False)
    original = argparse.ArgumentParser.parse_args
    parsed = {}

    class ParsingComplete(Exception):
        pass

    def capture_args(parser, *args, **kwargs):
        parsed.update(vars(original(parser, ["--ckpt", "example.ckpt"])))
        raise ParsingComplete

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_args)
    with pytest.raises(ParsingComplete):
        runpy.run_path(str(ROOT / "infer/eval_ckpt.py"), run_name="__main__")
    assert parsed["cache_dir"] == str(tmp_path / "cache/hub")
    assert parsed["data_root"] == str(tmp_path / "datasets")
    assert parsed["tok_len"] == 128


@pytest.mark.skipif(os.name != "posix" or not SHELL, reason="portable Bash interface runs on POSIX")
def test_batch_inference_can_forward_cache_options(tmp_path):
    executable = tmp_path / "fake evaluator"
    capture = tmp_path / "eval_args.json"
    executable.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "if sys.argv[1] == '-c':\n    print('summary')\n"
        "else:\n"
        "    with open(os.environ['CAPTURE'],'w') as handle:\n"
        "        json.dump(sys.argv[1:],handle)\n",
        encoding="utf-8")
    executable.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "ENV_PY": str(executable),
           "CAPTURE": str(capture)}
    extra = ["--cache-dir", str(tmp_path / "model cache"), "--data-root", str(tmp_path / "data cache")]
    result = subprocess.run([SHELL, str(ROOT / "infer/eval_all.sh"),
                             str(tmp_path / "out"), "rte:model.ckpt", "--", *extra],
                            env=env, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(capture.read_text())[-4:] == extra
