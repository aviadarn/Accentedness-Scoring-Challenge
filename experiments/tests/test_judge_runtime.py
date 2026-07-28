"""Tests for the isolated experimental judge runtime."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest import mock


RUNTIME_ROOT = (
    Path(__file__).resolve().parents[1] / "E10-local-llm-judges" / "runtime"
)
RUNTIME_SRC = RUNTIME_ROOT / "src"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

from judge_runtime.prepare_model import (  # noqa: E402
    DEFAULT_MODEL_ID,
    METADATA_FILENAME,
    prepare_model,
)
from judge_runtime.runner import (  # noqa: E402
    OFFLINE_ENVIRONMENT,
    PROBE_REQUEST_ID,
    load_mlx_backend,
    probe,
    serve,
)


class JudgeRuntimeTests(unittest.TestCase):
    def _model_snapshot(self, root: Path) -> Path:
        model_path = root / "model"
        model_path.mkdir()
        (model_path / "config.json").write_text(
            '{"model_type":"gemma3n"}\n', encoding="utf-8"
        )
        (model_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        return model_path

    def _audio(self, root: Path, name: str) -> Path:
        path = root / name
        path.write_bytes(b"RIFF-test-audio")
        return path

    def test_persistent_protocol_loads_once_and_uses_deterministic_audio_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = self._model_snapshot(root)
            first_audio = self._audio(root, "first.wav")
            second_audio = self._audio(root, "second.wav")

            load_calls: list[tuple[str, dict[str, object]]] = []
            template_calls: list[tuple[object, object, str, int]] = []
            generation_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            schema_calls: list[tuple[object, dict[str, object]]] = []
            offline_at_load: dict[str, str | None] = {}
            schema_processor = object()

            def fake_load(path: str, **kwargs: object) -> tuple[object, object]:
                print("loader chatter")
                load_calls.append((path, kwargs))
                offline_at_load.update(
                    {name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT}
                )
                return SimpleNamespace(config={"model_type": "gemma3n"}), object()

            def fake_template(
                processor: object,
                config: object,
                prompt: str,
                *,
                num_audios: int,
            ) -> str:
                template_calls.append((processor, config, prompt, num_audios))
                return f"templated:{prompt}:{num_audios}"

            generation_results: list[object] = [
                SimpleNamespace(text='{"score":1}'),
                '{"score":2}',
            ]

            def fake_generate(*args: object, **kwargs: object) -> object:
                print("generation chatter")
                generation_calls.append((args, kwargs))
                return generation_results.pop(0)

            def fake_schema_processor(tokenizer: object, schema: dict[str, object]):
                schema_calls.append((tokenizer, schema))
                return schema_processor

            def backend_factory(path: Path):
                return load_mlx_backend(
                    path,
                    load_fn=fake_load,
                    generate_fn=fake_generate,
                    apply_chat_template_fn=fake_template,
                    json_schema_processor_factory=fake_schema_processor,
                )

            requests = [
                {
                    "request_id": "first",
                    "audio_paths": [str(first_audio)],
                    "prompt": "Judge the first sample.",
                    "max_tokens": 64,
                    "response_format": "text",
                },
                {
                    "request_id": 2,
                    "audio_paths": [str(first_audio), str(second_audio)],
                    "prompt": "Judge both samples.",
                    "max_tokens": 128,
                    "response_format": "judge_json",
                },
            ]
            input_stream = io.StringIO(
                "".join(json.dumps(request) + "\n" for request in requests)
            )
            output_stream = io.StringIO()
            error_stream = io.StringIO()

            with mock.patch.dict(
                os.environ,
                {name: "0" for name in OFFLINE_ENVIRONMENT},
                clear=False,
            ):
                response_count = serve(
                    model_path,
                    input_stream=input_stream,
                    output_stream=output_stream,
                    error_stream=error_stream,
                    backend_factory=backend_factory,
                )

            self.assertEqual(response_count, 2)
            self.assertEqual(len(load_calls), 1)
            self.assertEqual(load_calls[0][0], str(model_path.resolve()))
            self.assertEqual(load_calls[0][1], {"local_files_only": True})
            self.assertEqual(offline_at_load, OFFLINE_ENVIRONMENT)
            self.assertEqual([call[3] for call in template_calls], [1, 2])
            self.assertEqual(len(generation_calls), 2)

            first_generation = generation_calls[0]
            self.assertEqual(first_generation[0][2], "templated:Judge the first sample.:1")
            self.assertEqual(
                first_generation[1]["audio"], [str(first_audio.resolve())]
            )
            self.assertEqual(first_generation[1]["temperature"], 0.0)
            self.assertEqual(first_generation[1]["max_tokens"], 64)
            self.assertIs(first_generation[1]["verbose"], False)
            self.assertNotIn("logits_processors", first_generation[1])
            self.assertEqual(
                generation_calls[1][1]["logits_processors"], [schema_processor]
            )
            self.assertEqual(len(schema_calls), 1)
            schema = schema_calls[0][1]
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
            self.assertEqual(schema["properties"]["phones"]["maxItems"], 100)
            self.assertEqual(schema["properties"]["notes"]["const"], "")
            phone_schema = schema["properties"]["phones"]["items"]
            self.assertEqual(phone_schema["properties"]["label"]["enum"], [0, 1, 2])

            responses = [
                json.loads(line) for line in output_stream.getvalue().splitlines()
            ]
            self.assertEqual(
                [list(response) for response in responses],
                [
                    ["request_id", "raw_text", "elapsed_seconds", "error"],
                    ["request_id", "raw_text", "elapsed_seconds", "error"],
                ],
            )
            self.assertEqual(
                [response["raw_text"] for response in responses],
                ['{"score":1}', '{"score":2}'],
            )
            self.assertEqual([response["error"] for response in responses], [None, None])
            self.assertIn("loader chatter", error_stream.getvalue())
            self.assertIn("generation chatter", error_stream.getvalue())
            self.assertNotIn("chatter", output_stream.getvalue())
            progress_lines = [
                line
                for line in error_stream.getvalue().splitlines()
                if " status=" in line
            ]
            self.assertEqual(len(progress_lines), 2)
            self.assertIn("request_id='first' status=success", progress_lines[0])
            self.assertIn("request_id=2 status=success", progress_lines[1])
            self.assertRegex(progress_lines[0], r"elapsed_seconds=\d+\.\d{3}$")

    def test_request_errors_are_correlated_and_do_not_stop_later_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = self._model_snapshot(root)
            audio_path = self._audio(root, "sample.wav")
            generated_ids: list[str | int] = []

            class FakeBackend:
                def generate(self, request):
                    generated_ids.append(request.request_id)
                    if request.request_id == "boom":
                        raise RuntimeError("synthetic generation failure")
                    return "recovered"

            lines = [
                "{not-json",
                json.dumps(
                    {
                        "request_id": "missing",
                        "audio_paths": [str(audio_path)],
                        "prompt": "prompt",
                        "response_format": "text",
                    }
                ),
                json.dumps(
                    {
                        "request_id": "remote",
                        "audio_paths": ["https://example.invalid/sample.wav"],
                        "prompt": "prompt",
                        "max_tokens": 12,
                        "response_format": "text",
                    }
                ),
                json.dumps(
                    {
                        "request_id": "format",
                        "audio_paths": [str(audio_path)],
                        "prompt": "prompt",
                        "max_tokens": 12,
                        "response_format": ["text"],
                    }
                ),
                json.dumps(
                    {
                        "request_id": "boom",
                        "audio_paths": [str(audio_path)],
                        "prompt": "prompt",
                        "max_tokens": 12,
                        "response_format": "text",
                    }
                ),
                json.dumps(
                    {
                        "request_id": "after",
                        "audio_paths": [str(audio_path)],
                        "prompt": "prompt",
                        "max_tokens": 12,
                        "response_format": "text",
                    }
                ),
            ]
            output_stream = io.StringIO()
            error_stream = io.StringIO()
            serve(
                model_path,
                input_stream=io.StringIO("\n".join(lines) + "\n"),
                output_stream=output_stream,
                error_stream=error_stream,
                backend_factory=lambda _path: FakeBackend(),
            )

            responses = [
                json.loads(line) for line in output_stream.getvalue().splitlines()
            ]
            self.assertEqual(
                [response["request_id"] for response in responses],
                [None, "missing", "remote", "format", "boom", "after"],
            )
            self.assertTrue(all(responses[index]["error"] for index in range(5)))
            self.assertIn("only local filesystem paths", responses[2]["error"])
            self.assertIn("response_format", responses[3]["error"])
            self.assertEqual(responses[5]["raw_text"], "recovered")
            self.assertIsNone(responses[5]["error"])
            self.assertEqual(generated_ids, ["boom", "after"])
            progress_lines = [
                line
                for line in error_stream.getvalue().splitlines()
                if " status=" in line
            ]
            self.assertEqual(len(progress_lines), 6)
            self.assertTrue(
                all("status=error" in line for line in progress_lines[:5])
            )
            self.assertIn("request_id='after' status=success", progress_lines[5])
            self.assertNotIn("https://example.invalid", "\n".join(progress_lines))
            self.assertNotIn("prompt", "\n".join(progress_lines))

    def test_blank_and_length_completions_are_correlated_and_runtime_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = self._model_snapshot(root)
            audio_path = self._audio(root, "sample.wav")
            generated = iter(
                (
                    SimpleNamespace(text="   ", finish_reason="stop"),
                    SimpleNamespace(text="partial output", finish_reason="length"),
                    SimpleNamespace(text="recovered", finish_reason="stop"),
                )
            )

            def backend_factory(path: Path):
                return load_mlx_backend(
                    path,
                    load_fn=lambda _path, **_kwargs: (
                        SimpleNamespace(config={"model_type": "gemma3n"}),
                        object(),
                    ),
                    generate_fn=lambda *_args, **_kwargs: next(generated),
                    apply_chat_template_fn=(
                        lambda _processor, _config, prompt, **_kwargs: prompt
                    ),
                )

            requests = [
                {
                    "request_id": request_id,
                    "audio_paths": [str(audio_path)],
                    "prompt": "prompt",
                    "max_tokens": 64,
                    "response_format": "text",
                }
                for request_id in ("blank", "length", "after")
            ]
            output_stream = io.StringIO()
            response_count = serve(
                model_path,
                input_stream=io.StringIO(
                    "".join(json.dumps(request) + "\n" for request in requests)
                ),
                output_stream=output_stream,
                error_stream=io.StringIO(),
                backend_factory=backend_factory,
            )

            responses = [
                json.loads(line) for line in output_stream.getvalue().splitlines()
            ]
            self.assertEqual(response_count, 3)
            self.assertEqual(
                [response["request_id"] for response in responses],
                ["blank", "length", "after"],
            )
            self.assertIn("blank text", responses[0]["error"])
            self.assertIn("exhausted max_tokens", responses[1]["error"])
            self.assertEqual(responses[2]["raw_text"], "recovered")
            self.assertIsNone(responses[2]["error"])

    def test_probe_loads_without_generation_and_rejects_remote_model_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = self._model_snapshot(Path(temporary_directory))
            load_count = 0

            class NeverGenerate:
                def generate(self, request):  # pragma: no cover - must never run
                    raise AssertionError(request)

            def backend_factory(_path: Path):
                nonlocal load_count
                load_count += 1
                return NeverGenerate()

            output_stream = io.StringIO()
            self.assertTrue(
                probe(
                    model_path,
                    output_stream=output_stream,
                    error_stream=io.StringIO(),
                    backend_factory=backend_factory,
                )
            )
            response = json.loads(output_stream.getvalue())
            self.assertEqual(load_count, 1)
            self.assertEqual(response["request_id"], PROBE_REQUEST_ID)
            self.assertIsNone(response["error"])

            remote_output = io.StringIO()
            self.assertFalse(
                probe(
                    DEFAULT_MODEL_ID,
                    output_stream=remote_output,
                    error_stream=io.StringIO(),
                    backend_factory=lambda _path: self.fail("must not load"),
                )
            )
            remote_response = json.loads(remote_output.getvalue())
            self.assertIn("local model path does not exist", remote_response["error"])

    def test_model_preparation_pins_commit_records_metadata_and_reuses_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "snapshot"
            commit_sha = "a" * 40
            model_info_calls: list[dict[str, object]] = []
            download_calls: list[dict[str, object]] = []

            def fake_model_info(**kwargs: object) -> object:
                model_info_calls.append(kwargs)
                return SimpleNamespace(sha=commit_sha.upper())

            def fake_snapshot_download(**kwargs: object) -> str:
                download_calls.append(kwargs)
                local_dir = Path(str(kwargs["local_dir"]))
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "config.json").write_text("{}\n", encoding="utf-8")
                (local_dir / "model.safetensors").write_bytes(b"weights")
                return str(local_dir)

            metadata = prepare_model(
                destination,
                model_info_fn=fake_model_info,
                snapshot_download_fn=fake_snapshot_download,
                log_stream=io.StringIO(),
            )

            self.assertEqual(
                model_info_calls,
                [{"repo_id": DEFAULT_MODEL_ID, "revision": "main"}],
            )
            self.assertEqual(len(download_calls), 1)
            self.assertEqual(download_calls[0]["revision"], commit_sha)
            self.assertEqual(download_calls[0]["repo_id"], DEFAULT_MODEL_ID)
            self.assertEqual(metadata["commit_sha"], commit_sha)
            self.assertEqual(metadata["runtime_dependency"], "mlx-vlm==0.6.8")
            self.assertEqual(metadata["snapshot_path"], str(destination.resolve()))
            self.assertEqual(
                json.loads((destination / METADATA_FILENAME).read_text(encoding="utf-8")),
                metadata,
            )

            reused = prepare_model(
                destination,
                model_info_fn=mock.Mock(side_effect=AssertionError("networked resolve")),
                snapshot_download_fn=mock.Mock(
                    side_effect=AssertionError("networked download")
                ),
                log_stream=io.StringIO(),
            )
            self.assertEqual(reused, metadata)

    def test_isolated_project_is_python_311_and_pins_mlx_vlm(self) -> None:
        project = tomllib.loads((RUNTIME_ROOT / "pyproject.toml").read_text("utf-8"))[
            "project"
        ]
        self.assertEqual(project["requires-python"], ">=3.11,<3.12")
        self.assertEqual(project["dependencies"], ["mlx-vlm==0.6.8"])


if __name__ == "__main__":
    unittest.main()
