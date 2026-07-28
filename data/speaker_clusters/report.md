# Pseudo-speaker analysis

Embedder: `microsoft/wavlm-base-plus-sv`. Linkage: average over cosine distance. Threshold: 0.92.

The manifests carry no speaker identifiers, so every speaker statement below is about clusters of similar voices, not verified speakers.

## Threshold calibration

Genuine same-voice pairs come from the two halves of one recording; impostor pairs come from halves of different recordings. Balancing the two gives an operating point, which is then carried onto the whole-clip similarity scale at a fixed false-accept rate.

| Distribution | Pairs | p5 | p50 | p95 |
|---|---:|---:|---:|---:|
| Genuine, half clips | 2894 | 0.7679 | 0.9228 | 0.9726 |
| Impostor, half clips | 14470 | 0.3043 | 0.6681 | 0.9119 |
| Impostor, whole clips | 14470 | 0.3539 | 0.7105 | 0.9356 |

Fragment length dominates verification accuracy, so the operating point is taken from the longest recordings that still supply enough pairs:

| Recordings of at least | Genuine pairs | Equal-error rate | Threshold |
|---:|---:|---:|---:|
| 0s | 2894 | 17.0% | 0.8509 |
| 3s | 1886 | 11.4% | 0.8767 |
| 4s | 1017 | 8.9% | 0.8895 |
| 5s (selected) | 416 | 8.0% | 0.8941 |

- calibrated on recordings of at least 5s (416 genuine pairs, 14470 impostor pairs): equal-error point 0.8941 at 8.0% error and 7.99% false accepts; holding that false-accept rate on whole recordings gives 0.9197

## Cluster structure

- 98 clusters over 3000 recordings
- Largest cluster 238, median size 2.0, singletons 37.8%
- Mean similarity within clusters 0.9504 versus 0.6654 between (separation 0.2849)

## Are the clusters voices or sentences?

- Two recordings in one cluster share a prompt 0.0% of the time, against a 0.1% dataset base rate (lift 0.12)
- Adjusted mutual information with prompt identity: -0.0221
- Non-singleton clusters containing more than one prompt: 100.0%

The clusters track voices, so the leakage numbers below are about speakers.

## Leakage in the shipped split

- 25 cluster(s) appear in both train and validation
- 97 of 100 validation recordings (97.0%) share a cluster with training
- 98.0% of validation phones sit in those recordings
- For comparison, prompt overlap is 92.0% of validation recordings

### Sensitivity to the threshold

| Similarity | Clusters | Shared | Record leakage | Phone leakage |
|---:|---:|---:|---:|---:|
| 0.80 | 9 | 4 | 99.0% | 99.7% |
| 0.81 | 10 | 5 | 99.0% | 99.7% |
| 0.82 | 12 | 6 | 99.0% | 99.7% |
| 0.83 | 13 | 7 | 99.0% | 99.7% |
| 0.84 | 15 | 8 | 99.0% | 99.7% |
| 0.85 | 18 | 11 | 99.0% | 99.7% |
| 0.86 | 21 | 12 | 99.0% | 99.7% |
| 0.87 | 26 | 13 | 99.0% | 99.7% |
| 0.88 | 36 | 13 | 98.0% | 99.2% |
| 0.89 | 41 | 15 | 98.0% | 99.2% |
| 0.90 | 55 | 17 | 97.0% | 98.0% |
| 0.91 | 81 | 22 | 97.0% | 98.0% |
| 0.92 | 100 | 25 | 97.0% | 98.0% |
| 0.93 | 140 | 29 | 97.0% | 98.0% |
| 0.94 | 211 | 37 | 95.0% | 96.7% |
| 0.95 | 320 | 46 | 93.0% | 95.6% |
| 0.96 | 537 | 51 | 88.0% | 92.2% |
| 0.97 | 944 | 65 | 81.0% | 87.8% |
| 0.98 | 1745 | 54 | 56.0% | 62.8% |
| 0.99 | 2823 | 11 | 11.0% | 13.9% |

## Recordings absent from both manifests

- 101 recording(s) across 28 cluster(s)
- 100 of them (99.0%) sit in a cluster that also holds labeled audio

## Speaker-disjoint replacement split

- Fit: 2534 recordings, 80102 phones, 26 clusters
- Dev: 365 recordings, 10137 phones, 71 clusters (11.2% of phones)
- Label mix (0/1/2): fit 12.2%/7.8%/79.9%, dev 12.5%/8.1%/79.4%
- Prompt overlap between the two sides: 87.1%. Speaker disjointness and prompt disjointness cut across each other in this dataset; this split enforces the first and reports the second.
