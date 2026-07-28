"""Pinned public artifacts and preprocessing constants for official GOPT."""

from __future__ import annotations


UPSTREAM_REPOSITORY = "https://github.com/YuanGongND/gopt"
UPSTREAM_COMMIT = "bed909daf8eca035095871e51642525acc5b9b55"
CHECKPOINT_RELATIVE_PATH = (
    "pretrained_models/gopt_librispeech/best_audio_model.pth"
)
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/YuanGongND/gopt/"
    f"{UPSTREAM_COMMIT}/{CHECKPOINT_RELATIVE_PATH}"
)
CHECKPOINT_SHA256 = (
    "ab07451e51648f9d2455505a51055b20ac4ad7921d771ccc5170ff486a826259"
)

OFFICIAL_DATA_ARCHIVE_PAGE = "https://share.weiyun.com/vJCAXjFY"
OFFICIAL_DATA_ARCHIVE_SHA256 = (
    "3cc533dd11eb273c60103b2cea076877170e3055df677d2f415769eff460ab17"
)

FEATURE_DIMENSION = 84
MAX_PHONE_COUNT = 50
FEATURE_MEAN = 3.203
FEATURE_STD = 4.045
FEATURE_SOURCE = "kaldi-gop-speechocean762-librispeech-m13"

MAPPING_VERSION = "gopt_speechocean762_librispeech_first_occurrence_v1"
PHONE_ID_ORDER = (
    "W",
    "IY",
    "K",
    "AO",
    "L",
    "IH",
    "T",
    "B",
    "EH",
    "R",
    "Z",
    "OW",
    "TH",
    "F",
    "AY",
    "V",
    "AH",
    "N",
    "UW",
    "S",
    "G",
    "AA",
    "M",
    "P",
    "NG",
    "HH",
    "EY",
    "SH",
    "AE",
    "D",
    "UH",
    "AW",
    "DH",
    "ER",
    "Y",
    "JH",
    "CH",
    "OY",
    "ZH",
)
PHONE_TO_ID = {phone: index for index, phone in enumerate(PHONE_ID_ORDER)}

MODEL_NAME = "official-gopt-librispeech"
SCORE_SCALE = "0-2"
SCORE_PROJECTION = "clip_0_2_v1"

