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

LOG = logging.getLogger(__name__)

def _eval_model(
    model: Union[str, lm_eval.api.model.LM],
    tasks: List[TaskConfiguration],
    model_args: Optional[Dict[str, Any]] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    **kwargs,
) -> Dict[str, Any]:
    # DEBUG: Log all incoming parameters
    import sys
    print(f"🔍 _eval_model called with:", file=sys.stderr)
    print(f"🔍   model: {model}", file=sys.stderr)
    print(f"🔍   model_args: {model_args}", file=sys.stderr)
    print(f"🔍   kwargs: {kwargs}", file=sys.stderr)
    
    # Create final model_args by merging and removing duplicates
    final_model_args = dict(model_args) if model_args else {}
    
    # Extract only safe lm_eval parameters from kwargs
    lm_eval_params = {}
    safe_params = ['num_fewshot', 'limit', 'apply_chat_template', 'fewshot_as_multiturn']
    
    for key in safe_params:
        if key in kwargs:
            lm_eval_params[key] = kwargs[key]
    
    # Handle batch_size separately to avoid conflicts
    if 'batch_size' in kwargs and 'batch_size' not in final_model_args:
        lm_eval_params['batch_size'] = kwargs['batch_size']
    
    print(f"🔍 Final parameters:", file=sys.stderr)
    print(f"🔍   final_model_args: {final_model_args}", file=sys.stderr)
    print(f"🔍   lm_eval_params: {lm_eval_params}", file=sys.stderr)
    
    results = lm_eval.simple_evaluate(
        model=model,
        model_args=final_model_args,
        tasks=list(set([task.name for task in tasks])),
        log_samples=False,
        verbosity="WARNING",
        task_manager=task_manager,
        **lm_eval_params,
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
    # monkeypatch_tqdm()
    monkeypatch_lmeval_vllm()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_args = {
            "pretrained": merged_path,
            "dtype": "bfloat16",
            **(model_kwargs or {}),
        }
        
        # Set device if not already specified, ensure it's a string
        if "device" not in model_args:
            model_args["device"] = device
        elif not isinstance(model_args["device"], str):
            model_args["device"] = str(model_args["device"])
        if vllm:
            model_args["gpu_memory_utilization"] = 0.8
            model_args["tensor_parallel_size"] = 1
            model_args["batch_size"] = batch_size or "auto"
            model_args["max_model_len"] = 4096
        else:
            model_args["use_cache"] = True

        # Remove any model-specific parameters from kwargs to avoid conflicts
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['device', 'dtype', 'pretrained', 'gpu_memory_utilization',
                                   'tensor_parallel_size', 'max_model_len', 'use_cache']}
        if 'batch_size' in clean_kwargs:
            del clean_kwargs['batch_size']

        LOG.info(f"clean_kwargs: {clean_kwargs}")

        res = _eval_model(
            "vllm" if vllm else "huggingface",
            tasks,
            model_args,
            num_fewshot=num_fewshot,
            limit=limit,
            task_manager=task_manager,
            **clean_kwargs,
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
