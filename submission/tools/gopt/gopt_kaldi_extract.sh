#!/usr/bin/env bash
# Run the official Kaldi SpeechOcean762 GOP front end on prepared challenge data.
#
# This script is intended to run inside the pinned Kaldi Linux container from
# GOPT_AUDIT.md, with the repository mounted at /workspace.  It refuses to
# reuse an output directory so a partial run cannot be mistaken for a complete
# extraction.

set -euo pipefail

usage() {
  echo "usage: $0 <prepared-data-dir> <m13-model-dir> <m13-ivector-dir> <lang-dir> <output-dir> [nj]" >&2
  exit 2
}

[[ $# -eq 5 || $# -eq 6 ]] || usage

prepared_data=$1
model_dir=$2
ivector_extractor=$3
lang_dir=$4
output_dir=$5
nj=${6:-4}

[[ $nj =~ ^[1-9][0-9]*$ ]] || {
  echo "gopt-kaldi-extract: nj must be a positive integer" >&2
  exit 2
}

for required in \
  "$prepared_data/wav.scp" \
  "$prepared_data/text" \
  "$prepared_data/text-phone" \
  "$prepared_data/utt2spk" \
  "$prepared_data/spk2utt" \
  "$model_dir/tree" \
  "$model_dir/final.mdl" \
  "$model_dir/phones.txt" \
  "$ivector_extractor/final.ie" \
  "$lang_dir/words.txt" \
  "$lang_dir/phones.txt" \
  "$lang_dir/phones/disambig.int"; do
  [[ -f $required ]] || {
    echo "gopt-kaldi-extract: required file is missing: $required" >&2
    exit 2
  }
done

[[ ! -e $output_dir ]] || {
  echo "gopt-kaldi-extract: output already exists: $output_dir" >&2
  exit 2
}

recipe_dir=${KALDI_GOP_RECIPE_DIR:-/opt/kaldi/egs/gop_speechocean762/s5}
[[ -f $recipe_dir/path.sh && -f $recipe_dir/conf/mfcc_hires.conf ]] || {
  echo "gopt-kaldi-extract: GOP recipe is unavailable: $recipe_dir" >&2
  exit 2
}

prepared_data=$(realpath "$prepared_data")
model_dir=$(realpath "$model_dir")
ivector_extractor=$(realpath "$ivector_extractor")
lang_dir=$(realpath "$lang_dir")
output_parent=$(realpath "$(dirname "$output_dir")")
output_dir=$output_parent/$(basename "$output_dir")

mkdir "$output_dir"
data_dir=$output_dir/data
log_dir=$output_dir/log
mfcc_dir=$output_dir/mfcc
ivector_dir=$output_dir/ivectors
prob_dir=$output_dir/probs
ali_dir=$output_dir/ali
gop_dir=$output_dir/gop
mkdir -p "$data_dir" "$log_dir" "$mfcc_dir" "$ali_dir/log" "$gop_dir/log"
cp \
  "$prepared_data/wav.scp" \
  "$prepared_data/text" \
  "$prepared_data/text-phone" \
  "$prepared_data/utt2spk" \
  "$prepared_data/spk2utt" \
  "$data_dir/"

cd "$recipe_dir"
source ./path.sh

utils/validate_data_dir.sh --no-feats "$data_dir"

steps/make_mfcc.sh \
  --nj "$nj" \
  --mfcc-config conf/mfcc_hires.conf \
  --cmd run.pl \
  "$data_dir" "$log_dir/make_mfcc" "$mfcc_dir"
steps/compute_cmvn_stats.sh \
  "$data_dir" "$log_dir/compute_cmvn" "$mfcc_dir"
utils/fix_data_dir.sh "$data_dir"

steps/online/nnet2/extract_ivectors_online.sh \
  --cmd run.pl \
  --nj "$nj" \
  "$data_dir" "$ivector_extractor" "$ivector_dir"

steps/nnet3/compute_output.sh \
  --cmd run.pl \
  --nj "$nj" \
  --online-ivector-dir "$ivector_dir" \
  "$data_dir" "$model_dir" "$prob_dir"

utils/split_data.sh "$data_dir" "$nj"
for job in $(seq 1 "$nj"); do
  utils/sym2int.pl -f 2- "$lang_dir/words.txt" \
    "$data_dir/split${nj}/${job}/text" \
    > "$data_dir/split${nj}/${job}/text.int"
done
utils/sym2int.pl -f 2- "$lang_dir/phones.txt" \
  "$data_dir/text-phone" > "$output_dir/text-phone.int"

run.pl JOB=1:"$nj" "$ali_dir/log/mk_align_graph.JOB.log" \
  compile-train-graphs-without-lexicon \
    --read-disambig-syms="$lang_dir/phones/disambig.int" \
    "$model_dir/tree" "$model_dir/final.mdl" \
    "ark,t:$data_dir/split${nj}/JOB/text.int" \
    "ark,t:$output_dir/text-phone.int" \
    "ark:|gzip -c > $ali_dir/fsts.JOB.gz"
printf '%s\n' "$nj" > "$ali_dir/num_jobs"

steps/align_mapped.sh \
  --cmd run.pl \
  --nj "$nj" \
  --graphs "$ali_dir" \
  "$data_dir" "$prob_dir" "$lang_dir" "$model_dir" "$ali_dir"

local/remove_phone_markers.pl \
  "$lang_dir/phones.txt" \
  "$gop_dir/phones-pure.txt" \
  "$gop_dir/phone-to-pure-phone.int"

run.pl JOB=1:"$nj" "$ali_dir/log/ali_to_phones.JOB.log" \
  ali-to-phones --per-frame=true "$model_dir/final.mdl" \
    "ark,t:gunzip -c $ali_dir/ali.JOB.gz|" \
    "ark,t:|gzip -c > $ali_dir/ali-phone.JOB.gz"

run.pl JOB=1:"$nj" "$gop_dir/log/compute_gop.JOB.log" \
  compute-gop \
    --phone-map="$gop_dir/phone-to-pure-phone.int" \
    --skip-phones-string=0:1:2 \
    "$model_dir/final.mdl" \
    "ark,t:gunzip -c $ali_dir/ali.JOB.gz|" \
    "ark,t:gunzip -c $ali_dir/ali-phone.JOB.gz|" \
    "ark:$prob_dir/output.JOB.ark" \
    "ark,t:$gop_dir/gop.JOB.txt" \
    "ark,t:$gop_dir/feat.JOB.txt"

cat "$gop_dir"/gop.*.txt > "$gop_dir/gop.txt"
cat "$gop_dir"/feat.*.txt > "$gop_dir/feat.txt"

sha256sum \
  "$model_dir/final.mdl" \
  "$model_dir/tree" \
  "$ivector_extractor/final.ie" \
  "$lang_dir/words.txt" \
  "$lang_dir/phones.txt" \
  "$gop_dir/phone-to-pure-phone.int" \
  "$gop_dir/feat.txt" \
  > "$output_dir/extraction-artifacts.sha256"

echo "gopt-kaldi-extract: wrote $gop_dir/feat.txt"
