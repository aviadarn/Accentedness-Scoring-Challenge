from __future__ import annotations

from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from accent_score.data import PHONE_VOCAB, PhoneRecord, load_manifest
from accent_score.judge_audit import (
    AuditError,
    AuditRunIncomplete,
    BlindTask,
    JudgeResult,
    JudgeRequest,
    ModelAuditResult,
    OllamaJudgeClient,
    PhoneAlignment,
    SubprocessJudgeClient,
    build_arg_parser,
    finalize_audit,
    load_blind_tasks,
    load_recheck_tasks,
    main as judge_audit_main,
    preflight_audit,
    prepare_audit,
    run_pass1,
    run_rechecks,
    select_anchor_records,
    select_disagreement_phones,
    validate_pass1,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPOSITORY_ROOT / "data" / "dataset"


def _write_wav(path: Path, *, frames: int = 1_600) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.zeros(frames, dtype=np.int16).tobytes())


def _make_dataset(root: Path, *, records_per_label: int = 2) -> Path:
    audio = root / "audio"
    audio.mkdir(parents=True)
    rows: list[dict] = []
    row_number = 0
    for label in (0, 1, 2):
        for offset in range(records_per_label):
            utterance_id = f"source_{row_number:03d}"
            _write_wav(audio / f"{utterance_id}.wav")
            phone = ("h", "s", "oʊ")[offset % 3]
            rows.append(
                {
                    "audio_path": f"audio/{utterance_id}.wav",
                    "text": f"secret source text {row_number}",
                    "phonemes": [{"phoneme": phone, "label": label}],
                }
            )
            row_number += 1
    (root / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return root


def _valid_judgment(task, *, label: int = 1) -> dict:
    return {
        "schema_version": 1,
        "audit_id": task.audit_id,
        "phones": [
            {
                "phone_index": index,
                "phoneme": phone,
                "label": label,
                "confidence": 0.8,
            }
            for index, phone in enumerate(task.phonemes)
        ],
        "notes": "independent pass",
    }


def test_actual_snapshot_selection_is_unique_balanced_and_vocab_distributed() -> None:
    records = load_manifest(
        DATASET_ROOT / "train.jsonl",
        dataset_root=DATASET_ROOT,
        validate_audio=False,
    )
    first = select_anchor_records(records, seed=42)
    second = select_anchor_records(records, seed=42)

    assert first == second
    assert len(first) == 150
    assert len({selection.manifest_row for selection in first}) == 150
    assert Counter(selection.anchor_label for selection in first) == {0: 50, 1: 50, 2: 50}
    phones_by_label: dict[int, set[str]] = defaultdict(set)
    for selection in first:
        phones_by_label[selection.anchor_label].add(selection.anchor_phoneme)
    assert {label: len(phones) for label, phones in phones_by_label.items()} == {
        0: 44,
        1: 44,
        2: 44,
    }
    assert {selection.audit_id for selection in first} == {
        f"A{index:04d}" for index in range(1, 151)
    }


def test_prepare_packet_is_anonymous_strict_and_byte_identical(tmp_path: Path) -> None:
    data_root = _make_dataset(tmp_path / "data")
    audit_root = tmp_path / "audit"
    summary = prepare_audit(
        data_root,
        audit_root,
        records_per_label=2,
        verify_snapshot=False,
    )
    tasks = load_blind_tasks(audit_root)

    assert len(tasks) == 6
    assert summary["item_count"] == 6
    assert Counter(item["anchor_label"] for item in summary["items"]) == {
        0: 2,
        1: 2,
        2: 2,
    }
    serialized = (audit_root / "blind" / "tasks.jsonl").read_text(encoding="utf-8")
    assert "label" not in serialized
    assert "score" not in serialized
    assert "source_" not in serialized
    assert str(tmp_path) not in serialized
    for task in tasks:
        assert set(task.to_dict()) == {
            "schema_version",
            "audit_id",
            "audio_path",
            "text",
            "phonemes",
        }
        assert task.audio_path == f"audio/{task.audit_id}.wav"
        private = next(
            item for item in summary["items"] if item["audit_id"] == task.audit_id
        )
        source = data_root / "audio" / f"{private['utterance_id']}.wav"
        assert (audit_root / "blind" / task.audio_path).read_bytes() == source.read_bytes()


def test_judge_schema_rejects_wrong_order_extra_fields_and_invalid_values() -> None:
    task = BlindTask("A0001", "audio/A0001.wav", "hello", ("h", "oʊ"))
    valid = _valid_judgment(task)
    parsed = JudgeResult.from_dict(valid, task=task)
    assert [phone.phoneme for phone in parsed.phones] == ["h", "oʊ"]

    wrong_order = json.loads(json.dumps(valid))
    wrong_order["phones"][0]["phone_index"] = 1
    with pytest.raises(AuditError, match="contiguous"):
        JudgeResult.from_dict(wrong_order, task=task)
    extra = json.loads(json.dumps(valid))
    extra["model_score"] = 99
    with pytest.raises(AuditError, match="fields must be exactly"):
        JudgeResult.from_dict(extra, task=task)
    invalid_label = json.loads(json.dumps(valid))
    invalid_label["phones"][0]["label"] = True
    with pytest.raises(AuditError, match="integer"):
        JudgeResult.from_dict(invalid_label, task=task)
    invalid_confidence = json.loads(json.dumps(valid))
    invalid_confidence["phones"][0]["confidence"] = 1.1
    with pytest.raises(AuditError, match=r"within \[0, 1\]"):
        JudgeResult.from_dict(invalid_confidence, task=task)


def test_pass1_retries_persists_and_resumes_without_rejudging(tmp_path: Path) -> None:
    data_root = _make_dataset(tmp_path / "data")
    audit_root = tmp_path / "audit"
    prepare_audit(data_root, audit_root, records_per_label=2, verify_snapshot=False)
    calls: Counter[str] = Counter()

    def flaky(request):
        calls[request.audit_id] += 1
        task = next(task for task in load_blind_tasks(audit_root) if task.audit_id == request.audit_id)
        if request.audit_id == "A0001" and calls[request.audit_id] == 1:
            return "not json"
        return _valid_judgment(task)

    summary = run_pass1(audit_root, flaky, max_retries=2)
    assert summary == {
        "tasks": 6,
        "already_complete": 0,
        "newly_complete": 6,
        "complete": 6,
    }
    assert calls["A0001"] == 2
    assert len(validate_pass1(audit_root)) == 6

    def must_not_run(_request):
        raise AssertionError("completed task was judged again")

    resumed = run_pass1(audit_root, must_not_run)
    assert resumed["already_complete"] == resumed["complete"] == 6
    assert resumed["newly_complete"] == 0


def test_ollama_client_is_decoupled_and_sends_only_blind_inputs(tmp_path: Path) -> None:
    audio = tmp_path / "A0001.wav"
    _write_wav(audio)
    captured: dict = {}

    def opener(request, *, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data)
        envelope = {"message": {"content": '{"schema_version": 1}'}}
        return io.BytesIO(json.dumps(envelope).encode())

    client = OllamaJudgeClient(
        model="local-audio-model",
        base_url="http://127.0.0.1:9999/",
        timeout_seconds=12,
        opener=opener,
    )
    content = client(
        JudgeRequest(
            "A0001",
            audio,
            "no sir",
            ("n", "oʊ", "s", "ɝ"),
            attempt=2,
            prior_error="phone count mismatch",
        )
    )

    assert content == '{"schema_version": 1}'
    assert captured["url"] == "http://127.0.0.1:9999/api/chat"
    assert captured["timeout"] == 12
    body = captured["body"]
    assert body["model"] == "local-audio-model"
    assert body["options"]["seed"] == 43
    assert body["options"]["num_predict"] == 4096
    prompt = body["messages"][0]["content"]
    assert "phone count mismatch" in prompt
    assert "dataset_label" not in prompt
    assert "model_score" not in prompt


class _FakeRuntimePipe:
    def __init__(self, runtime, mode: str) -> None:
        self.runtime = runtime
        self.mode = mode
        self.closed = False

    def write(self, text: str) -> int:
        assert self.mode == "stdin"
        self.runtime.request = json.loads(text)
        return len(text)

    def flush(self) -> None:
        return None

    def readline(self) -> str:
        assert self.mode == "stdout"
        response = self.runtime.handler(self.runtime.request)
        return json.dumps(response) + "\n"

    def close(self) -> None:
        self.closed = True


class _FakeRuntimeProcess:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.request = None
        self.stdin = _FakeRuntimePipe(self, "stdin")
        self.stdout = _FakeRuntimePipe(self, "stdout")
        self.waited = False
        self.terminated = False

    def poll(self):
        return 0 if self.waited else None

    def wait(self, *, timeout):
        assert timeout > 0
        self.waited = True
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True


def test_subprocess_client_uses_persistent_strict_ndjson_protocol(tmp_path: Path) -> None:
    audio = tmp_path / "A0001.wav"
    model = tmp_path / "model"
    model.mkdir()
    _write_wav(audio)
    launches: list[tuple[list[str], dict]] = []

    def handler(request):
        return {
            "request_id": request["request_id"],
            "raw_text": '{"schema_version":1}',
            "elapsed_seconds": 0.25,
            "error": None,
        }

    process = _FakeRuntimeProcess(handler)

    def popen(argv, **kwargs):
        launches.append((argv, kwargs))
        return process

    request = JudgeRequest("A0001", audio, "no sir", ("n", "oʊ"), attempt=2)
    with SubprocessJudgeClient(
        model,
        command=("fake-python", "runtime.py"),
        popen_factory=popen,
    ) as client:
        first = client(request)
        second = client(request)

    assert first == second == '{"schema_version":1}'
    assert len(launches) == 1
    assert launches[0][0] == [
        "fake-python",
        "runtime.py",
        "--model-path",
        str(model.resolve()),
    ]
    payload = process.request
    assert set(payload) == {
        "request_id",
        "audio_paths",
        "prompt",
        "max_tokens",
        "response_format",
    }
    assert payload["request_id"] == "judge:A0001:2"
    assert payload["audio_paths"] == [str(audio.resolve())]
    assert payload["max_tokens"] == 4096
    assert payload["response_format"] == "judge_json"
    assert "no sir" in payload["prompt"]
    assert process.stdin.closed
    assert process.waited


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda response: response.update(request_id="wrong"), "request_id"),
        (lambda response: response.update(error="generation failed"), "generation failed"),
    ],
)
def test_subprocess_client_rejects_mismatch_and_runtime_error(
    tmp_path: Path, mutate, match: str
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    audio = tmp_path / "A0001.wav"
    _write_wav(audio)

    def handler(request):
        response = {
            "request_id": request["request_id"],
            "raw_text": "",
            "elapsed_seconds": 0.1,
            "error": None,
        }
        mutate(response)
        return response

    process = _FakeRuntimeProcess(handler)
    client = SubprocessJudgeClient(
        model,
        command=("runtime",),
        popen_factory=lambda _argv, **_kwargs: process,
    )
    with client, pytest.raises(RuntimeError, match=match):
        client.generate(
            request_id="request-1",
            audio_paths=(audio,),
            prompt="transcribe",
            max_tokens=32,
            response_format="text",
        )


def test_subprocess_client_relaunches_after_transport_death(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    audio = tmp_path / "A0001.wav"
    _write_wav(audio)

    first = _FakeRuntimeProcess(lambda _request: {})
    first.stdout.readline = lambda: ""
    second = _FakeRuntimeProcess(
        lambda request: {
            "request_id": request["request_id"],
            "raw_text": "recovered",
            "elapsed_seconds": 0.1,
            "error": None,
        }
    )
    processes = iter((first, second))
    launches = 0

    def popen(_argv, **_kwargs):
        nonlocal launches
        launches += 1
        return next(processes)

    client = SubprocessJudgeClient(
        model,
        command=("runtime",),
        popen_factory=popen,
    )
    with pytest.raises(RuntimeError, match="before replying"):
        client.generate(
            request_id="first",
            audio_paths=(audio,),
            prompt="transcribe",
            max_tokens=32,
            response_format="text",
        )
    assert client.generate(
        request_id="second",
        audio_paths=(audio,),
        prompt="transcribe",
        max_tokens=32,
        response_format="text",
    ) == "recovered"
    client.close()
    assert launches == 2


def test_retry_exhaustion_keeps_prior_rows_for_resume(tmp_path: Path) -> None:
    data_root = _make_dataset(tmp_path / "data", records_per_label=1)
    audit_root = tmp_path / "audit"
    prepare_audit(data_root, audit_root, records_per_label=1, verify_snapshot=False)
    tasks = load_blind_tasks(audit_root)

    def client(request):
        if request.audit_id == tasks[1].audit_id:
            return {}
        task = next(task for task in tasks if task.audit_id == request.audit_id)
        return _valid_judgment(task)

    with pytest.raises(AuditRunIncomplete, match="safely persisted"):
        run_pass1(audit_root, client, max_retries=2)
    persisted = (audit_root / "ratings" / "pass1.jsonl").read_text().splitlines()
    assert len(persisted) == 1
    with pytest.raises(AuditError, match="incomplete"):
        validate_pass1(audit_root)


def test_preflight_is_audio_only_gates_5_plus_10_and_seeds_resume(
    tmp_path: Path,
) -> None:
    data_root = _make_dataset(tmp_path / "data", records_per_label=4)
    audit_root = tmp_path / "audit"
    prepare_audit(data_root, audit_root, records_per_label=4, verify_snapshot=False)
    tasks = load_blind_tasks(audit_root)
    task_by_id = {task.audit_id: task for task in tasks}

    class PreflightClient:
        def __init__(self) -> None:
            self.structured_calls = 0
            self.transcript_ids: list[str] = []

        def generate(
            self, *, request_id, audio_paths, prompt, max_tokens, response_format
        ):
            assert request_id.startswith("preflight-transcript:")
            assert len(audio_paths) == 1
            assert max_tokens == 64
            assert response_format == "text"
            audit_id = Path(audio_paths[0]).stem
            task = task_by_id[audit_id]
            assert task.text not in prompt
            self.transcript_ids.append(audit_id)
            if len(self.transcript_ids) == 1:
                return ""
            return task.text.upper() + "!"

        def __call__(self, request):
            self.structured_calls += 1
            if self.structured_calls == 1:
                return "not-json"
            return _valid_judgment(
                task_by_id[request.audit_id], label=self.structured_calls % 2
            )

    preflight_client = PreflightClient()
    summary = preflight_audit(audit_root, preflight_client)

    assert summary["passed"] is True
    assert summary["transcription"]["tasks"] == 5
    assert summary["transcription"]["nonempty"] == 4
    assert summary["transcription"]["median_word_error_rate"] == 0.0
    assert summary["structured"]["tasks"] == 10
    assert summary["structured"]["valid"] == 9
    assert summary["structured"]["distinct_predicted_labels"] == 2
    assert summary["structured"]["single_label_share"] <= 0.95
    assert preflight_client.structured_calls == 10
    assert len(set(preflight_client.transcript_ids)) == 5
    assert len((audit_root / "ratings" / "pass1.jsonl").read_text().splitlines()) == 9

    remaining_calls: list[str] = []

    def finish(request):
        remaining_calls.append(request.audit_id)
        return _valid_judgment(task_by_id[request.audit_id])

    resumed = run_pass1(audit_root, finish)
    assert resumed["already_complete"] == 9
    assert resumed["newly_complete"] == 3
    assert len(remaining_calls) == 3


def test_preflight_skips_structured_work_when_audio_gate_fails(
    tmp_path: Path,
) -> None:
    data_root = _make_dataset(tmp_path / "data", records_per_label=4)
    audit_root = tmp_path / "audit"
    prepare_audit(data_root, audit_root, records_per_label=4, verify_snapshot=False)

    class FailedAudioClient:
        transcript_calls = 0

        def generate(
            self, *, request_id, audio_paths, prompt, max_tokens, response_format
        ):
            del request_id, audio_paths, prompt, max_tokens, response_format
            self.transcript_calls += 1
            return ""

        def __call__(self, request):  # pragma: no cover - must be gated off
            raise AssertionError(f"structured judge unexpectedly called for {request.audit_id}")

    client = FailedAudioClient()
    with pytest.raises(AuditError, match="structured gate skipped"):
        preflight_audit(audit_root, client)

    summary = json.loads(
        (audit_root / "private" / "preflight.json").read_text(encoding="utf-8")
    )
    assert client.transcript_calls == 5
    assert summary["passed"] is False
    assert summary["transcription"]["nonempty"] == 0
    assert summary["structured"]["tasks"] == 0
    assert summary["structured"]["skipped_due_to_transcription_gate"] is True
    assert not (audit_root / "private" / "pass1_attempts.jsonl").exists()


def test_failed_structured_preflight_commits_no_rows_and_blocks_cli_run(
    tmp_path: Path,
) -> None:
    data_root = _make_dataset(tmp_path / "data", records_per_label=4)
    audit_root = tmp_path / "audit"
    prepare_audit(data_root, audit_root, records_per_label=4, verify_snapshot=False)
    tasks = load_blind_tasks(audit_root)
    task_by_id = {task.audit_id: task for task in tasks}

    class InvalidStructuredClient:
        def generate(
            self, *, request_id, audio_paths, prompt, max_tokens, response_format
        ):
            del request_id, prompt, max_tokens, response_format
            return task_by_id[Path(audio_paths[0]).stem].text

        def __call__(self, request):
            return {
                "schema_version": 1,
                "audit_id": request.audit_id,
                "phones": [],
                "notes": "",
            }

    with pytest.raises(AuditError, match="structured valid=0/10"):
        preflight_audit(audit_root, InvalidStructuredClient())

    assert not (audit_root / "ratings" / "pass1.jsonl").exists()
    with pytest.raises(AuditError, match="latest gate did not pass"):
        judge_audit_main(
            ["run", "--audit-dir", str(audit_root)],
            judge_client=InvalidStructuredClient(),
        )


def test_preflight_rejects_structurally_valid_single_label_collapse(
    tmp_path: Path,
) -> None:
    data_root = _make_dataset(tmp_path / "data", records_per_label=4)
    audit_root = tmp_path / "audit"
    prepare_audit(data_root, audit_root, records_per_label=4, verify_snapshot=False)
    tasks = load_blind_tasks(audit_root)
    task_by_id = {task.audit_id: task for task in tasks}

    class CollapsedClient:
        def generate(
            self, *, request_id, audio_paths, prompt, max_tokens, response_format
        ):
            del request_id, prompt, max_tokens, response_format
            return task_by_id[Path(audio_paths[0]).stem].text

        def __call__(self, request):
            return _valid_judgment(task_by_id[request.audit_id], label=2)

    with pytest.raises(AuditError, match="predicted labels=1"):
        preflight_audit(audit_root, CollapsedClient())

    summary = json.loads(
        (audit_root / "private" / "preflight.json").read_text(encoding="utf-8")
    )
    assert summary["structured"]["valid"] == 10
    assert summary["structured"]["predicted_label_counts"] == {
        "0": 0,
        "1": 0,
        "2": 10,
    }
    assert summary["structured"]["single_label_share"] == 1.0
    assert summary["passed"] is False
    assert not (audit_root / "ratings" / "pass1.jsonl").exists()


def test_disagreement_selection_is_deterministic_limited_and_label_balanced() -> None:
    items: list[dict] = []
    for label in (0, 1, 2):
        for index in range(100):
            items.append(
                {
                    "audit_id": f"A{label}{index:03d}",
                    "phone_index": 0,
                    "phoneme": PHONE_VOCAB[index % len(PHONE_VOCAB)],
                    "dataset_label": label,
                    "judge_label": (label + 1) % 3,
                    "judge_confidence": 0.9,
                    "model_score": 50.0,
                    "flags": ["judge_dataset_disagreement"],
                }
            )
    first = select_disagreement_phones(items, limit=200, seed=42)
    second = select_disagreement_phones(items, limit=200, seed=42)
    by_id = {item["audit_id"]: item for item in items}
    counts = Counter(by_id[audit_id]["dataset_label"] for audit_id, _ in first)

    assert first == second
    assert len(first) == 200
    assert max(counts.values()) - min(counts.values()) <= 1
    with pytest.raises(ValueError, match="through 200"):
        select_disagreement_phones(items, limit=201)


def _complete_judging(audit_root: Path) -> None:
    task_by_id = {task.audit_id: task for task in load_blind_tasks(audit_root)}

    def client(request):
        numeric = int(request.audit_id[1:])
        return _valid_judgment(task_by_id[request.audit_id], label=numeric % 3)

    run_pass1(audit_root, client)


def test_finalize_unblinds_metrics_items_rechecks_and_clip_specs(tmp_path: Path) -> None:
    data_root = _make_dataset(tmp_path / "data")
    audit_root = tmp_path / "audit"
    prepare_audit(data_root, audit_root, records_per_label=2, verify_snapshot=False)
    _complete_judging(audit_root)
    first_task = load_blind_tasks(audit_root)[0]
    rechecks = audit_root / "ratings" / "rechecks.jsonl"
    rechecks.write_text(
        json.dumps(
            {
                "audit_id": first_task.audit_id,
                "phone_index": 0,
                "label": 2,
                "confidence": 0.95,
                "notes": "clip recheck",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, list[str]]] = []

    def model_runner(path: str, phones: list[str]) -> ModelAuditResult:
        calls.append((path, phones))
        return ModelAuditResult(
            tuple(20.0 + 30.0 * index for index in range(len(phones))),
            tuple(PhoneAlignment(index, index + 1) for index in range(len(phones))),
        )

    report = finalize_audit(
        data_root,
        audit_root,
        model_runner,
        rechecks_path=rechecks,
        n_bootstrap=5,
        disagreement_limit=3,
        verify_snapshot=False,
    )
    item_rows = [
        json.loads(line)
        for line in (audit_root / "report" / "items.jsonl").read_text().splitlines()
    ]
    clips = [
        json.loads(line)
        for line in (audit_root / "report" / "clips.jsonl").read_text().splitlines()
    ]

    assert len(calls) == 6
    assert report["provenance"]["anchor_label_counts"] == {0: 2, 1: 2, 2: 2}
    assert report["overall"]["phones"] == 6
    assert set(report["overall"]) >= {
        "model_vs_dataset",
        "model_vs_judge",
        "judge_vs_dataset",
    }
    assert len(item_rows) == 6
    required = {
        "audit_id",
        "utterance_id",
        "audio_path",
        "text",
        "phone_index",
        "phoneme",
        "dataset_label",
        "judge_label",
        "judge_confidence",
        "model_score",
        "model_class",
        "recheck_label",
        "recheck_confidence",
        "recheck_notes",
        "alignment",
        "clip",
        "flags",
    }
    assert required <= set(item_rows[0])
    assert item_rows[0]["recheck_label"] == 2
    assert len(clips) <= 3
    assert report["disagreements"]["selected"] <= 3
    for clip in clips:
        assert 0.0 <= clip["start_seconds"] < clip["end_seconds"] <= 0.1
        assert clip["suggested_output_path"].startswith("clips/A")


def test_finalize_writes_pcm16_blind_tasks_and_recheck_run_round_trip(
    tmp_path: Path,
) -> None:
    data_root = _make_dataset(tmp_path / "data")
    audit_root = tmp_path / "audit"
    prepare_audit(data_root, audit_root, records_per_label=2, verify_snapshot=False)
    _complete_judging(audit_root)

    def model_runner(_path: str, phones: list[str]) -> ModelAuditResult:
        return ModelAuditResult(
            tuple(100.0 for _ in phones),
            tuple(PhoneAlignment(1, 2) for _ in phones),
        )

    report = finalize_audit(
        data_root,
        audit_root,
        model_runner,
        n_bootstrap=2,
        disagreement_limit=3,
        verify_snapshot=False,
    )
    recheck_tasks = load_recheck_tasks(audit_root)
    serialized_tasks = (audit_root / "blind" / "recheck_tasks.jsonl").read_text()

    assert len(recheck_tasks) == report["disagreements"]["pcm16_clips"]
    assert len(recheck_tasks) > 0
    assert "dataset_label" not in serialized_tasks
    assert "judge_label" not in serialized_tasks
    assert "model_score" not in serialized_tasks
    for task in recheck_tasks:
        with wave.open(str(audit_root / task.audio_path), "rb") as reader:
            assert reader.getsampwidth() == 2
            assert reader.getframerate() == 16_000
            assert reader.getnframes() == 1_600

    seen_prompts: list[str] = []

    def recheck_client(request):
        seen_prompts.append(request.text)
        assert request.text == ""
        return {
            "schema_version": 1,
            "audit_id": request.audit_id,
            "phones": [
                {
                    "phone_index": 0,
                    "phoneme": request.phonemes[0],
                    "label": 2,
                    "confidence": 0.91,
                }
            ],
            "notes": "clip-only recheck",
        }

    recheck_summary = run_rechecks(audit_root, recheck_client)
    assert recheck_summary["complete"] == len(recheck_tasks)
    rows = [
        json.loads(line)
        for line in (audit_root / "ratings" / "rechecks.jsonl").read_text().splitlines()
    ]
    assert all(
        set(row) == {"audit_id", "phone_index", "label", "confidence", "notes"}
        for row in rows
    )
    assert len(seen_prompts) == len(recheck_tasks)

    finalize_audit(
        data_root,
        audit_root,
        model_runner,
        n_bootstrap=2,
        disagreement_limit=3,
        verify_snapshot=False,
    )
    final_items = [
        json.loads(line)
        for line in (audit_root / "report" / "items.jsonl").read_text().splitlines()
    ]
    rechecked_keys = {(row["audit_id"], row["phone_index"]) for row in rows}
    assert all(
        item["recheck_label"] == 2
        for item in final_items
        if (item["audit_id"], item["phone_index"]) in rechecked_keys
    )
    assert all(
        item["clip"]["padding_seconds"] == 0.30
        for item in final_items
        if item["clip"] is not None
    )


def test_finalize_rejects_tampered_blind_packet_before_model_call(tmp_path: Path) -> None:
    data_root = _make_dataset(tmp_path / "data", records_per_label=1)
    audit_root = tmp_path / "audit"
    prepare_audit(data_root, audit_root, records_per_label=1, verify_snapshot=False)
    _complete_judging(audit_root)
    tasks_path = audit_root / "blind" / "tasks.jsonl"
    tasks_path.write_text(tasks_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(AuditError, match="fingerprint"):
        finalize_audit(
            data_root,
            audit_root,
            lambda _path, _phones: (_ for _ in ()).throw(
                AssertionError("model must not run")
            ),
            n_bootstrap=2,
            verify_snapshot=False,
        )


def test_cli_defaults_to_mlx_and_requires_explicit_legacy_ollama() -> None:
    parser = build_arg_parser()
    mlx = parser.parse_args(
        [
            "run",
            "--audit-dir",
            "audit",
            "--judge-model-path",
            "judge-model",
        ]
    )
    ollama = parser.parse_args(
        [
            "run",
            "--audit-dir",
            "audit",
            "--judge-backend",
            "ollama",
        ]
    )

    assert mlx.judge_backend == "mlx"
    assert mlx.runtime_command
    assert ollama.judge_backend == "ollama"
