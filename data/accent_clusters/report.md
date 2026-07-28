# Accent-pattern clustering

This analysis groups pseudo-speakers by **which phones differ**, after removing
each speaker's overall accentedness component. It does not cluster WavLM vectors.

## Result

- Selected clusters: 4
- Supported pseudo-speakers: 97 of 98
- Assigned recordings: 2999 of 3000
- Silhouette: 0.207
- Resampling stability (mean ARI): 0.822

## Cluster descriptions

- Cluster 0: 4 pseudo-speakers; overall accentedness 0.217; distinctive phones: ɝ (more accented), aar (more accented), ɛ (less accented), ɹ (less accented), aor (more accented)
- Cluster 1: 3 pseudo-speakers; overall accentedness 0.201; distinctive phones: ɝ (less accented), z (more accented), v (more accented), aar (less accented), ɪ (more accented)
- Cluster 2: 84 pseudo-speakers; overall accentedness 0.147; distinctive phones: ɝ (less accented), ɑ (less accented), ɹ (less accented), aar (less accented), ʊ (more accented)
- Cluster 3: 6 pseudo-speakers; overall accentedness 0.152; distinctive phones: ɹ (more accented), ɝ (more accented), ð (less accented), ɑ (more accented), z (less accented)

## Caveats

- Pseudo-speakers are inferred from audio and are not verified speaker identities.
- Accent clusters are unsupervised pronunciation-pattern groups, not nationality, language, or ethnicity labels.
- Cluster numbers have no ordinal meaning and must not be interpreted as better or worse accents.
- Sparse phones are shrunk toward corpus phone means; reliability matrices must be consulted before interpreting individuals.
- Centroids are fit only on well-supported pseudo-speakers; sparse speakers receive explicitly provisional, evidence-discounted assignments.
- Train and validation labels are combined only for exploratory clustering, not for model evaluation.
- Unreferenced recordings inherit a cluster only when their pseudo-speaker has labeled recordings.
