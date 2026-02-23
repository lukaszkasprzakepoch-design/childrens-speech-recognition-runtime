#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

TOKEN_RE = re.compile(r"[a-zA-Z']+")

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "at",
    "is",
    "it",
    "its",
    "i",
    "you",
    "he",
    "she",
    "they",
    "we",
    "my",
    "your",
    "me",
    "this",
    "that",
    "these",
    "those",
    "are",
    "was",
    "were",
    "be",
    "am",
    "been",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "with",
    "as",
    "by",
    "from",
    "but",
    "if",
    "then",
    "so",
    "because",
    "um",
    "uh",
    "yeah",
    "oh",
    "hmm",
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    k = (len(values) - 1) * (p / 100.0)
    floor = math.floor(k)
    ceil = math.ceil(k)
    if floor == ceil:
        return values[floor]
    return values[floor] + (values[ceil] - values[floor]) * (k - floor)


def describe_numeric(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    sorted_values = sorted(float(v) for v in values)
    mean_v = statistics.fmean(sorted_values)
    median_v = statistics.median(sorted_values)
    std_v = statistics.stdev(sorted_values) if len(sorted_values) > 1 else 0.0

    return {
        "mean": mean_v,
        "median": median_v,
        "std": std_v,
        "p95": percentile(sorted_values, 95),
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def write_bar(
    labels: list[str],
    values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    rotate_x: bool = False,
    color: str = "#4C78A8",
) -> None:
    plt.figure(figsize=(12, 7))
    plt.bar(labels, values, color=color)
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def write_hist(
    values: list[float | int],
    bins: int,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    color: str = "#4C78A8",
) -> None:
    plt.figure(figsize=(12, 7))
    plt.hist(values, bins=bins, color=color, edgecolor="black")
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def summarize(input_jsonl: Path, output_dir: Path, top_k: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    age_counter: Counter[str] = Counter()
    clips_per_child: Counter[str] = Counter()
    clips_per_session: Counter[str] = Counter()
    words_counter: Counter[str] = Counter()
    content_words_counter: Counter[str] = Counter()

    durations: list[float] = []
    words_per_clip: list[int] = []
    chars_per_clip: list[int] = []

    words_by_age: dict[str, list[int]] = defaultdict(list)
    duration_by_age: dict[str, list[float]] = defaultdict(list)

    short_clips_lt_1s = 0
    long_clips_gt_30s = 0
    very_long_clips_gt_60s = 0
    dense_transcripts_gt_100_words = 0
    empty_transcripts = 0

    total_clips = 0

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            total_clips += 1

            age = row.get("age_bucket", "unknown")
            child_id = row.get("child_id", "unknown")
            session_id = row.get("session_id", "unknown")
            text = (row.get("orthographic_text") or "").strip()
            duration = float(row.get("audio_duration_sec") or 0.0)

            tokens = tokenize(text)
            token_count = len(tokens)
            char_count = len(text)

            age_counter[age] += 1
            clips_per_child[child_id] += 1
            clips_per_session[session_id] += 1

            words_counter.update(tokens)
            content_words_counter.update(t for t in tokens if t not in STOPWORDS)

            durations.append(duration)
            words_per_clip.append(token_count)
            chars_per_clip.append(char_count)
            words_by_age[age].append(token_count)
            duration_by_age[age].append(duration)

            if token_count == 0:
                empty_transcripts += 1
            if duration < 1.0:
                short_clips_lt_1s += 1
            if duration > 30.0:
                long_clips_gt_30s += 1
            if duration > 60.0:
                very_long_clips_gt_60s += 1
            if token_count > 100:
                dense_transcripts_gt_100_words += 1

    age_labels = sorted(age_counter.keys(), key=lambda x: (x != "unknown", x))
    age_counts = [age_counter[a] for a in age_labels]

    avg_words_by_age = [
        statistics.fmean(words_by_age[a]) if words_by_age[a] else 0.0 for a in age_labels
    ]
    avg_duration_by_age = [
        statistics.fmean(duration_by_age[a]) if duration_by_age[a] else 0.0 for a in age_labels
    ]

    top_words = words_counter.most_common(top_k)
    top_content_words = content_words_counter.most_common(top_k)

    write_bar(
        age_labels,
        [float(v) for v in age_counts],
        "Clip Count by Age Bucket",
        "Age Bucket",
        "Number of Clips",
        output_dir / "age_distribution.png",
    )

    write_hist(
        words_per_clip,
        30,
        "Words per Clip Distribution",
        "Words per Clip",
        "Number of Clips",
        output_dir / "words_per_clip_histogram.png",
    )

    write_hist(
        durations,
        40,
        "Duration Distribution",
        "Duration (seconds)",
        "Number of Clips",
        output_dir / "duration_histogram.png",
    )

    write_hist(
        list(clips_per_child.values()),
        40,
        "Clips per Child Distribution",
        "Clips per Child",
        "Number of Children",
        output_dir / "clips_per_child_histogram.png",
        color="#59A14F",
    )

    write_bar(
        age_labels,
        avg_words_by_age,
        "Average Words per Clip by Age",
        "Age Bucket",
        "Average Words",
        output_dir / "avg_words_by_age.png",
    )

    write_bar(
        age_labels,
        avg_duration_by_age,
        "Average Duration by Age",
        "Age Bucket",
        "Average Duration (seconds)",
        output_dir / "avg_duration_by_age.png",
    )

    if top_content_words:
        words, freqs = zip(*top_content_words)
        write_bar(
            list(words),
            [float(v) for v in freqs],
            f"Top {len(top_content_words)} Content Words",
            "Word",
            "Frequency",
            output_dir / "top_content_words.png",
            rotate_x=True,
            color="#E15759",
        )

    total_tokens = sum(words_counter.values())
    vocab_size = len(words_counter)
    type_token_ratio = (vocab_size / total_tokens) if total_tokens else 0.0

    summary = {
        "num_clips": total_clips,
        "num_children": len(clips_per_child),
        "num_sessions": len(clips_per_session),
        "age_distribution": dict(sorted(age_counter.items())),
        "duration_sec": describe_numeric(durations),
        "words_per_clip": describe_numeric(words_per_clip),
        "chars_per_clip": describe_numeric(chars_per_clip),
        "clips_per_child": describe_numeric(list(clips_per_child.values())),
        "clips_per_session": describe_numeric(list(clips_per_session.values())),
        "vocabulary": {
            "total_tokens": total_tokens,
            "unique_tokens": vocab_size,
            "type_token_ratio": type_token_ratio,
            "top_words": top_words,
            "top_content_words": top_content_words,
        },
        "data_quality_signals": {
            "empty_transcripts": empty_transcripts,
            "short_clips_lt_1s": short_clips_lt_1s,
            "long_clips_gt_30s": long_clips_gt_30s,
            "very_long_clips_gt_60s": very_long_clips_gt_60s,
            "dense_transcripts_gt_100_words": dense_transcripts_gt_100_words,
        },
    }

    (output_dir / "extended_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    (output_dir / "top_content_words.tsv").write_text(
        "word\tfrequency\n" + "\n".join(f"{w}\t{c}" for w, c in top_content_words),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate extended label statistics and plots for transcript JSONL files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/train_word_transcripts.jsonl"),
        help="Input transcript JSONL path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("statisitcs/plots_extended"),
        help="Directory where plots and summary are written",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="How many top words to include in outputs",
    )

    args = parser.parse_args()
    summary = summarize(args.input, args.output_dir, args.top_k)
    print(json.dumps(summary, indent=2))
    print(f"\nExtended plots written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
