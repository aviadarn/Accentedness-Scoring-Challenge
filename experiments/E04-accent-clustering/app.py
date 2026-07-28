#!/usr/bin/env python3
"""Browse pronunciation-pattern clusters and listen to their recordings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_ROOT = EXPERIMENTS_ROOT / "_support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

from bootstrap import REPOSITORY_ROOT, bootstrap_imports

bootstrap_imports()

import gradio as gr
import pandas as pd


DEFAULT_CLUSTER_DIR = REPOSITORY_ROOT / "data/accent_clusters"
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data/dataset"


class AccentClusterExplorerError(ValueError):
    """Raised when generated clustering artifacts are incomplete or unsafe."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AccentClusterExplorerError(f"cannot read {path}: {error}") from error


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise AccentClusterExplorerError(f"cannot read {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise AccentClusterExplorerError(f"blank JSONL row at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AccentClusterExplorerError(
                f"invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise AccentClusterExplorerError(
                f"JSONL row at {path}:{line_number} must be an object"
            )
        rows.append(row)
    if not rows:
        raise AccentClusterExplorerError(f"artifact is empty: {path}")
    return tuple(rows)


def pattern_name(cluster: int) -> str:
    """Return a compact deterministic display name for a numeric cluster."""

    if isinstance(cluster, bool) or not isinstance(cluster, int) or cluster < 0:
        raise AccentClusterExplorerError("accent cluster must be a non-negative integer")
    value = cluster
    suffix = ""
    while True:
        value, remainder = divmod(value, 26)
        suffix = chr(ord("A") + remainder) + suffix
        if value == 0:
            return f"Pattern {suffix}"
        value -= 1


def _safe_audio_path(data_dir: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AccentClusterExplorerError("recording audio_path must be a relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise AccentClusterExplorerError(f"unsafe recording audio path: {relative!r}")
    candidate = data_dir.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise AccentClusterExplorerError(f"recording audio is missing: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(data_dir)
    except ValueError as error:
        raise AccentClusterExplorerError(
            f"recording audio escapes the dataset: {relative!r}"
        ) from error
    return resolved


@dataclass(frozen=True, slots=True)
class ExplorerData:
    report: Mapping[str, Any]
    speakers: tuple[Mapping[str, Any], ...]
    recordings: tuple[Mapping[str, Any], ...]
    data_dir: Path

    @property
    def selected_k(self) -> int:
        value = self.report.get("selected_k")
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise AccentClusterExplorerError("report selected_k must be at least two")
        return value

    @property
    def cluster_ids(self) -> tuple[int, ...]:
        observed = {
            row["accent_cluster"]
            for row in self.speakers
            if row.get("accent_cluster") is not None
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in observed):
            raise AccentClusterExplorerError("speaker accent clusters must be integers")
        expected = set(range(self.selected_k))
        if observed != expected:
            raise AccentClusterExplorerError(
                f"speaker clusters are {sorted(observed)}, expected {sorted(expected)}"
            )
        return tuple(sorted(observed))

    def summary_for(self, cluster: int) -> Mapping[str, Any]:
        summaries = self.report.get("cluster_summaries")
        if not isinstance(summaries, list):
            raise AccentClusterExplorerError("report cluster_summaries must be an array")
        for summary in summaries:
            if isinstance(summary, Mapping) and summary.get("accent_cluster") == cluster:
                return summary
        return {}

    def recordings_for(self, cluster: int) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            row for row in self.recordings if row.get("accent_cluster") == cluster
        )

    def recording(self, audio_path: str) -> Mapping[str, Any]:
        for row in self.recordings:
            if row.get("audio_path") == audio_path:
                return row
        raise AccentClusterExplorerError(f"unknown recording: {audio_path}")


def load_explorer_data(
    cluster_dir: str | Path = DEFAULT_CLUSTER_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> ExplorerData:
    """Load and cross-check the five generated explorer artifacts."""

    cluster_root = Path(cluster_dir).expanduser().resolve()
    dataset_root = Path(data_dir).expanduser().resolve()
    if not cluster_root.is_dir() or cluster_root.is_symlink():
        raise AccentClusterExplorerError(
            f"cluster artifact directory does not exist: {cluster_root}"
        )
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise AccentClusterExplorerError(f"dataset directory does not exist: {dataset_root}")
    report = _load_json(cluster_root / "report.json")
    if not isinstance(report, Mapping):
        raise AccentClusterExplorerError("report.json must contain one object")
    speakers = _load_jsonl(cluster_root / "speakers.jsonl")
    recordings = _load_jsonl(cluster_root / "recordings.jsonl")
    speaker_ids: set[int] = set()
    for row in speakers:
        speaker = row.get("speaker_cluster")
        if isinstance(speaker, bool) or not isinstance(speaker, int) or speaker < 0:
            raise AccentClusterExplorerError("speaker_cluster must be a non-negative integer")
        if speaker in speaker_ids:
            raise AccentClusterExplorerError(f"duplicate speaker row: {speaker}")
        speaker_ids.add(speaker)
    audio_paths: set[str] = set()
    for row in recordings:
        relative = row.get("audio_path")
        _safe_audio_path(dataset_root, relative)
        if relative in audio_paths:
            raise AccentClusterExplorerError(f"duplicate recording row: {relative}")
        audio_paths.add(relative)
        if row.get("speaker_cluster") not in speaker_ids:
            raise AccentClusterExplorerError(
                f"recording references unknown speaker: {row.get('speaker_cluster')}"
            )
    data = ExplorerData(
        report=report,
        speakers=speakers,
        recordings=recordings,
        data_dir=dataset_root,
    )
    data.cluster_ids
    return data


def _phone_list(values: Any) -> str:
    if not isinstance(values, list):
        return "Not reported"
    rendered: list[str] = []
    for item in values[:8]:
        if not isinstance(item, Mapping):
            continue
        phone = item.get("phone", "?")
        direction = str(item.get("direction", "different")).replace("_", " ")
        delta = item.get("pattern_delta")
        detail = f"{float(delta):+.3f}" if isinstance(delta, (int, float)) else ""
        rendered.append(f"`{phone}` ({direction} {detail})".strip())
    return ", ".join(rendered) or "Not reported"


def _cluster_markdown(data: ExplorerData, cluster: int) -> str:
    summary = data.summary_for(cluster)
    rows = data.recordings_for(cluster)
    speaker_count = len(
        {
            int(row["speaker_cluster"])
            for row in rows
            if isinstance(row.get("speaker_cluster"), int)
        }
    )
    fit_speakers = summary.get("fit_speaker_count")
    provisional_speakers = summary.get("provisional_speaker_count")
    evidence_text = (
        f"{int(fit_speakers)} centroid-fit + {int(provisional_speakers)} provisional"
        if isinstance(fit_speakers, int) and isinstance(provisional_speakers, int)
        else f"{speaker_count} provisional voices"
    )
    labeled = sum(bool(row.get("labeled")) for row in rows)
    severity = summary.get("overall_accentedness")
    if severity is None:
        severity = summary.get("mean_accentedness")
    severity_text = (
        f"{100.0 * float(severity):.1f}%"
        if isinstance(severity, (int, float)) and float(severity) <= 1.0
        else (f"{float(severity):.1f}%" if isinstance(severity, (int, float)) else "n/a")
    )
    phones = summary.get("top_distinctive_phones")
    if phones is None:
        matching = [row for row in data.speakers if row.get("accent_cluster") == cluster]
        phones = matching[0].get("top_distinctive_phones") if matching else None
    return (
        f"## {pattern_name(cluster)}\n\n"
        f"**{speaker_count} pseudo-voices ({evidence_text}) · {len(rows)} recordings · "
        f"{labeled} labeled · mean accentedness {severity_text}**\n\n"
        f"Most distinctive phone directions: {_phone_list(phones)}\n\n"
        "These are anonymous pronunciation patterns, not nationality or native-language labels."
    )


def _recording_choices(
    data: ExplorerData, cluster: int
) -> tuple[list[tuple[str, str]], list[list[Any]]]:
    rows = sorted(
        data.recordings_for(cluster),
        key=lambda row: (
            int(row.get("speaker_cluster", -1)),
            str(row.get("audio_path", "")),
        ),
    )
    choices: list[tuple[str, str]] = []
    table: list[list[Any]] = []
    for row in rows:
        audio_path = str(row["audio_path"])
        sentence = str(row.get("text") or "Unlabeled extra take")
        short_sentence = sentence if len(sentence) <= 72 else sentence[:69] + "..."
        speaker = int(row["speaker_cluster"])
        choices.append((f"{Path(audio_path).stem} · voice {speaker} · {short_sentence}", audio_path))
        severity = row.get("mean_accentedness")
        table.append(
            [
                Path(audio_path).stem,
                speaker,
                row.get("split"),
                row.get("assignment_status"),
                None if severity is None else round(100.0 * float(severity), 1),
                short_sentence,
            ]
        )
    return choices, table[:250]


def _recording_markdown(row: Mapping[str, Any]) -> str:
    sentence = str(row.get("text") or "No transcript: unreferenced extra take")
    severity = row.get("mean_accentedness")
    severity_text = (
        "unavailable" if severity is None else f"{100.0 * float(severity):.1f}%"
    )
    return (
        f"**Voice cluster:** {row['speaker_cluster']}  \n"
        f"**Dataset split:** {row.get('split')}  \n"
        f"**Assignment evidence:** {row.get('assignment_status')}  \n"
        f"**Recording mean accentedness:** {severity_text}  \n"
        f"**Sentence:** {sentence}"
    )


def _scatter_frame(data: ExplorerData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for speaker in data.speakers:
        cluster = speaker.get("accent_cluster")
        if cluster is None:
            continue
        rows.append(
            {
                "x": float(speaker["x"]),
                "y": float(speaker["y"]),
                "pattern": pattern_name(int(cluster)),
                "voice": str(speaker["speaker_cluster"]),
                "confidence": round(float(speaker.get("assignment_confidence", 0.0)), 3),
                "labeled recordings": int(speaker.get("labeled_recordings", 0)),
            }
        )
    return pd.DataFrame(rows)


def build_explorer(
    cluster_dir: str | Path = DEFAULT_CLUSTER_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> gr.Blocks:
    """Build the local explorer without starting a server."""

    data = load_explorer_data(cluster_dir, data_dir)
    initial_cluster = data.cluster_ids[0]
    initial_choices, initial_table = _recording_choices(data, initial_cluster)
    initial_audio = initial_choices[0][1]
    initial_row = data.recording(initial_audio)
    cluster_choices = [
        (f"{pattern_name(cluster)} ({len(data.recordings_for(cluster))} recordings)", cluster)
        for cluster in data.cluster_ids
    ]

    def select_cluster(cluster_value: Any):
        cluster = int(cluster_value)
        if cluster not in data.cluster_ids:
            raise gr.Error("Unknown pronunciation-pattern cluster", print_exception=False)
        choices, table = _recording_choices(data, cluster)
        first = choices[0][1]
        row = data.recording(first)
        return (
            _cluster_markdown(data, cluster),
            gr.update(choices=choices, value=first),
            table,
            str(_safe_audio_path(data.data_dir, first)),
            _recording_markdown(row),
        )

    def select_recording(audio_path: str):
        row = data.recording(audio_path)
        return (
            str(_safe_audio_path(data.data_dir, audio_path)),
            _recording_markdown(row),
        )

    with gr.Blocks(title="Pronunciation-pattern cluster explorer") as app:
        gr.Markdown("# Pronunciation-pattern cluster explorer")
        gr.Markdown(
            "The 3,000 recordings are first grouped by provisional voice, then clustered "
            "from phone-level accentedness patterns. Overall strength is removed before "
            "clustering. Cluster names are anonymous because the dataset has no country "
            "or native-language metadata."
        )
        gr.ScatterPlot(
            value=_scatter_frame(data),
            x="x",
            y="y",
            color="pattern",
            title="Speaker-level pronunciation profiles",
            x_title="Profile component 1",
            y_title="Profile component 2",
            tooltip=["voice", "pattern", "confidence", "labeled recordings"],
            height=430,
            buttons=["fullscreen", "export"],
        )
        cluster = gr.Dropdown(
            choices=cluster_choices,
            value=initial_cluster,
            label="Pronunciation pattern",
            interactive=True,
        )
        summary = gr.Markdown(_cluster_markdown(data, initial_cluster))
        recording = gr.Dropdown(
            choices=initial_choices,
            value=initial_audio,
            label="Recording to hear",
            interactive=True,
        )
        audio = gr.Audio(
            value=str(_safe_audio_path(data.data_dir, initial_audio)),
            type="filepath",
            label="Selected recording",
            interactive=False,
            autoplay=False,
        )
        recording_details = gr.Markdown(_recording_markdown(initial_row))
        table = gr.Dataframe(
            value=initial_table,
            headers=[
                "Recording",
                "Voice",
                "Split",
                "Assignment",
                "Accentedness %",
                "Sentence",
            ],
            datatype=["str", "number", "str", "str", "number", "str"],
            interactive=False,
            wrap=True,
            label="Recordings in this pattern (first 250)",
        )

        cluster.change(
            fn=select_cluster,
            inputs=[cluster],
            outputs=[summary, recording, table, audio, recording_details],
        )
        recording.change(
            fn=select_recording,
            inputs=[recording],
            outputs=[audio, recording_details],
        )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        app = build_explorer(args.cluster_dir, args.data_dir)
    except AccentClusterExplorerError as error:
        raise SystemExit(f"accent-cluster-app: error: {error}") from error
    app.launch(server_name=args.host, server_port=args.port, share=False, show_error=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
