# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Union

import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface
import lm_eval.tasks
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch

from mergekit.evo.config import TaskConfiguration
from mergekit.evo.genome import InvalidGenotypeError, ModelGenome
from mergekit.evo.monkeypatch import monkeypatch_lmeval_vllm
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


def _eval_model(
    model: Union[str, lm_eval.api.model.LM],
    tasks: List[TaskConfiguration],
    model_args: Optional[Dict[str, Any]] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    **kwargs,
) -> Dict[str, Any]:
    # DEBUG: Log all parameters to identify device conflicts
    logging.warning(f"DEBUG _eval_model - model_args: {model_args}")
    logging.warning(f"DEBUG _eval_model - kwargs: {kwargs}")
    
    # Remove device from kwargs to avoid conflict with model_args
    eval_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
    logging.warning(f"DEBUG _eval_model - eval_kwargs after device removal: {eval_kwargs}")
    
    results = lm_eval.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=list(set([task.name for task in tasks])),
        log_samples=False,
        verbosity="WARNING",
        task_manager=task_manager,
        **eval_kwargs,
    )

    logging.info(results["results"])
    res = 0
    for task in tasks:
        res += results["results"][task.name][task.metric] * task.weight
    return {"score": res, "results": results["results"]}


def evaluate_model(
    merged_path: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    vllm: bool,
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> dict:
    # DEBUG: Log initial parameters
    logging.warning(f"DEBUG evaluate_model - model_kwargs: {model_kwargs}")
    logging.warning(f"DEBUG evaluate_model - kwargs: {kwargs}")
    
    # monkeypatch_tqdm()
    monkeypatch_lmeval_vllm()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_args = {
            "pretrained": merged_path,
            "dtype": "bfloat16",
            **(model_kwargs or {}),
        }
        logging.warning(f"DEBUG evaluate_model - model_args after creation: {model_args}")
        
        # Set device if not already specified, ensure it's a string
        if "device" not in model_args:
            model_args["device"] = device
        elif not isinstance(model_args["device"], str):
            model_args["device"] = str(model_args["device"])
        
        logging.warning(f"DEBUG evaluate_model - final model_args: {model_args}")
        if vllm:
            model_args["gpu_memory_utilization"] = 0.8
            model_args["tensor_parallel_size"] = 1
            model_args["batch_size"] = "auto"
            model_args["max_model_len"] = 4096
        else:
            model_args["use_cache"] = True

        # Remove device from kwargs to avoid duplicate parameter
        eval_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        
        res = _eval_model(
            "vllm" if vllm else "huggingface",
            tasks,
            model_args,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            task_manager=task_manager,
            **eval_kwargs,
        )
        return res
    finally:
        shutil.rmtree(merged_path)


evaluate_model_ray = ray.remote(num_cpus=1, num_gpus=1.0)(evaluate_model)


def merge_model(
    genotype: torch.Tensor,
    genome: ModelGenome,
    model_storage_path: str,
    merge_options: MergeOptions,
) -> str:
    # monkeypatch_tqdm()
    try:
        cfg = genome.genotype_merge_config(genotype)
    except InvalidGenotypeError as e:
        logging.error("Invalid genotype", exc_info=e)
        return None
    os.makedirs(model_storage_path, exist_ok=True)
    res = tempfile.mkdtemp(prefix="merged", dir=model_storage_path)
    run_merge(cfg, out_path=res, options=merge_options)
    return res


merge_model_ray = ray.remote(
    num_cpus=1,
    num_gpus=1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model)
