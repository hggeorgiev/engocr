"""CLI tests: config bootstrap, convert output conventions (fake provider)."""

import json
import os

import pytest
from PIL import Image

import engocr.cli as cli


class FakeProvider:
    def analyze(self, prompt, image):
        return json.dumps({
            "page_summary": "summary",
            "text_elements": [{"text": "hello world", "type": "handwriting"}],
        })


@pytest.fixture(autouse=True)
def fake_extractor(monkeypatch):
    from engocr.extractor import VisionExtractor as RealExtractor
    monkeypatch.setattr(cli, "VisionExtractor",
                        lambda: RealExtractor(provider=FakeProvider()))


def _image(tmp_path, name):
    path = tmp_path / name
    Image.new("RGB", (8, 8)).save(path)
    return path


# ── config bootstrap ──────────────────────────────────


def test_config_init_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGOCR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("VISION_PROVIDER", raising=False)

    path = cli.init_config()
    data = json.loads(path.read_text())
    data["api"]["gemini_key"] = "sk-test"
    data["vision_provider"] = "anthropic"
    path.write_text(json.dumps(data))

    cli.load_user_config()
    assert os.environ["GEMINI_API_KEY"] == "sk-test"
    assert os.environ["VISION_PROVIDER"] == "anthropic"

    # env wins over config.json
    monkeypatch.setenv("VISION_PROVIDER", "openai")
    cli.load_user_config()
    assert os.environ["VISION_PROVIDER"] == "openai"


def test_config_init_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGOCR_CONFIG_DIR", str(tmp_path / "cfg"))
    cli.init_config()
    with pytest.raises(FileExistsError):
        cli.init_config()


# ── convert output conventions ────────────────────────


def test_convert_writes_md_next_to_input(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ENGOCR_CONFIG_DIR", str(tmp_path / "none"))
    img = _image(tmp_path, "lecture.jpg")

    cli.main(["convert", str(img)])

    out = tmp_path / "lecture.md"
    assert out.exists()
    text = out.read_text()
    assert text.startswith("# lecture")
    assert "hello world" in text
    assert "→" in capsys.readouterr().out


def test_convert_stdout(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ENGOCR_CONFIG_DIR", str(tmp_path / "none"))
    img = _image(tmp_path, "a.png")

    cli.main(["convert", str(img), "--stdout"])

    assert "# a" in capsys.readouterr().out
    assert not (tmp_path / "a.md").exists()


def test_convert_json_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGOCR_CONFIG_DIR", str(tmp_path / "none"))
    img = _image(tmp_path, "b.png")

    cli.main(["convert", str(img), "--json"])

    data = json.loads((tmp_path / "b.json").read_text())
    assert data["pages"][0]["page_summary"] == "summary"


def test_convert_out_dir_multiple_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGOCR_CONFIG_DIR", str(tmp_path / "none"))
    imgs = [_image(tmp_path, n) for n in ("x.jpg", "y.jpg")]
    out_dir = tmp_path / "out"

    cli.main(["convert", *(str(i) for i in imgs), "-o", str(out_dir)])

    assert (out_dir / "x.md").exists()
    assert (out_dir / "y.md").exists()


def test_convert_missing_file_errors_but_continues(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ENGOCR_CONFIG_DIR", str(tmp_path / "none"))
    img = _image(tmp_path, "ok.jpg")

    with pytest.raises(SystemExit) as e:
        cli.main(["convert", str(tmp_path / "nope.jpg")])
    assert e.value.code == 1

    cli.main(["convert", str(tmp_path / "nope.jpg"), str(img)])
    assert (tmp_path / "ok.md").exists()
    assert "Not found" in capsys.readouterr().err
