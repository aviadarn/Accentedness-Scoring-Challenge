# E14 train-only pseudo-speaker artifact

This artifact is for grouped model selection, not the E03 leakage audit. Pseudo-speakers are provisional voice clusters, not verified identities.

## Validated artifact declarations

- Manifest: `train.jsonl` (2799 recordings; SHA-256 `f6650855bf62ebbec1e1a60cb8fb491d0e5fb0fb20667d402299fc1238a8148b`)
- Artifact SHA-256: `f4d46c32c0a879828a95c43c113c9ffa4bf42cdba13dd337d59bbb73d192533a`
- E14 independently recomputes the manifest hash, recording-key hash, row count, and exact row membership when loading this artifact.
- The process-scope fields below are generator declarations; they are schema-validated provenance, not an independent attestation of which files the generator opened.
- Calibration scope: `train-manifest-recordings-only`
- Clustering scope: `train-manifest-recordings-only`
- `val.jsonl` loaded: no
- Validation or unreferenced audio loaded: no
- Non-training embedding vectors used for fit: no
- Training keys were selected before calibration and linkage: yes

The declared cache-reuse boundary relies on pretrained embedding inference being independent per recording. Non-training vectors may exist in those caches; the generator declares that it selected training keys before fitting any statistic or tree.

- Whole cache rows: 3000 total, 2799 selected
- Half cache rows: 2995 total, 2794 selected

## Grouping summary

- Embedder: `microsoft/wavlm-base-plus-sv`
- Similarity threshold: 0.9183
- Linkage: `average`
- Provisional groups: 89
- Largest group: 226; median size: 2.0; singleton fraction: 39.3%
- Within/between similarity separation: 0.2839
- Prompt-text lift inside groups: 0.132 (required <= 1.500)
