import argparse
import io
import json
import logging
import os
import zipfile
from functools import cache

import datasets
import numpy as np
import requests
try:
    from lm_eval.tasks.longbench.metrics import (
        # classification_score,
        code_sim_score,
        count_score,
        qa_f1_score,
        qa_f1_zh_score,
        retrieval_score,
        retrieval_zh_score,
        rouge_score,
        rouge_zh_score,
    )
except ModuleNotFoundError:
    # Fallback for direct execution from this directory.
    from metrics import (
        # classification_score,
        code_sim_score,
        count_score,
        qa_f1_score,
        qa_f1_zh_score,
        retrieval_score,
        retrieval_zh_score,
        rouge_score,
        rouge_zh_score,
    )


eval_logger = logging.getLogger(__name__)
LONG_BENCH_DATA_ZIP_URL = (
    "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
)


@cache
def _download_longbench_zip(dataset_url: str) -> bytes:
    response = requests.get(dataset_url, timeout=120)
    response.raise_for_status()
    return response.content


def _load_longbench_records(dataset_name: str, dataset_url: str) -> list[dict]:
    zip_bytes = _download_longbench_zip(dataset_url)
    member_path = f"data/{dataset_name}.jsonl"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        if member_path not in zf.namelist():
            raise ValueError(
                f"Could not find {member_path} in LongBench archive from {dataset_url}."
            )
        with zf.open(member_path) as fp:
            return [json.loads(line) for line in fp if line.strip()]


def load_dataset(**kwargs):
    dataset_name = kwargs.get("dataset_name")
    if not dataset_name:
        raise ValueError("LongBench custom loader requires dataset_kwargs.dataset_name")

    split = kwargs.get("split", "test")
    dataset_url = kwargs.get("dataset_url", LONG_BENCH_DATA_ZIP_URL)
    eval_logger.info(
        "Loading LongBench dataset %s from %s into split '%s'",
        dataset_name,
        dataset_url,
        split,
    )
    records = _load_longbench_records(dataset_name=dataset_name, dataset_url=dataset_url)
    return {split: datasets.Dataset.from_list(records)}


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    # "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    # "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# def parse_args(args=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, default=None)
#     parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
#     return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)
