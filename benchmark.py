import argparse
import random
from itertools import chain
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import DFlashDraftModel, load_and_process_dataset
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--entropy-tree", action="store_true", default=False)
    parser.add_argument("--draft-temperature", type=float, default=1.0)
    parser.add_argument("--log-entropy", action="store_true", default=False)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()
    if not installed_flash_attn:
        raise RuntimeError("flash_attn must be installed because the draft DFlash model always uses FlashAttention")

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = "flash_attention_2"

    if not args.flash_attn and installed_flash_attn:
        logger.warning("DDTree uses a custom tree attention mask on the target model. For compatibility, forcing the target verifier to torch.sdpa.")

    # Single-process path: split both models across all available GPUs via device_map="auto".
    # This handles GPU memory constraints where a single GPU can't hold both models (e.g. 2×16GB
    # GPUs with two 8B models). Each model gets ~half the memory per GPU, leaving headroom for
    # KV cache and activations.
    # Multi-process path (data-parallel): each rank loads the full models onto its own GPU.
    _use_model_parallel = dist.size() == 1 and torch.cuda.device_count() > 1
    if _use_model_parallel:
        n_gpus = torch.cuda.device_count()
        # Reserve ~7.2 GiB per GPU for the target model; draft uses whatever remains.
        _target_max_mem = {i: "7200MiB" for i in range(n_gpus)}
        target = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=target_attn_implementation,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=_target_max_mem,
        ).eval()
        draft_model = DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=draft_attn_implementation,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
    else:
        target = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=target_attn_implementation,
            dtype=torch.bfloat16,
        ).to(device).eval()
        draft_model = DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=draft_attn_implementation,
            dtype=torch.bfloat16,
        ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
    methods_to_run = ["dflash"]
    method_key_to_tree_budget = {}
    if not args.flash_attn:
        if args.entropy_tree:
            ddtree_method_keys = [f"ddtree_entropy_tb{b}" for b in tree_budgets]
        elif args.draft_temperature != 1.0:
            temp_str = f"{args.draft_temperature:.1f}".replace(".", "p")
            ddtree_method_keys = [f"ddtree_temp{temp_str}_tb{b}" for b in tree_budgets]
        else:
            ddtree_method_keys = [f"ddtree_tb{b}" for b in tree_budgets]
        methods_to_run.extend(ddtree_method_keys)
        method_key_to_tree_budget.update(dict(zip(ddtree_method_keys, tree_budgets)))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.sample_seed if args.sample_seed is not None else 0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = dflash_generate(
        model=draft_model,
        target=target,
        input_ids=warmup_input_ids,
        mask_token_id=draft_model.mask_token_id,
        max_new_tokens=warmup_max_new_tokens,
        block_size=1,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
    )
    for method_key in methods_to_run:
        if method_key == "dflash":
            _ = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
        else:
            _ = ddtree_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                tree_budget=method_key_to_tree_budget[method_key],
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                entropy_tree=args.entropy_tree,
                draft_temperature=args.draft_temperature,
            )

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            response["baseline"] = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=1,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
            for method_key in methods_to_run:
                if method_key == "dflash":
                    response[method_key] = dflash_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                else:
                    response[method_key] = ddtree_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        tree_budget=method_key_to_tree_budget[method_key],
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                        entropy_tree=args.entropy_tree,
                        draft_temperature=args.draft_temperature,
                        log_entropy=args.log_entropy,
                    )

            spec_response = response[methods_to_run[-1]]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    run_data = {
        "responses": responses,
        "block_size": block_size,
        "draft_attn_implementation": draft_attn_implementation,
        "target_attn_implementation": target_attn_implementation,
        "args": vars(args),
    }
    
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run_data, save_path)

    if args.log_entropy and args.save_path is not None:
        import csv
        ddtree_method = next((m for m in methods_to_run if m.startswith("ddtree_")), None)
        if ddtree_method is not None:
            entropy_save_path = Path(args.save_path).with_suffix(".entropy.csv")
            with open(entropy_save_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_idx", "round_idx", "depth_idx", "H_j", "log_P_star_j"])
                for sample_idx, resp in enumerate(responses):
                    if ddtree_method in resp and resp[ddtree_method].entropy_logs is not None:
                        for round_idx, round_log in enumerate(resp[ddtree_method].entropy_logs):
                            for depth_idx, (h, lp) in enumerate(zip(round_log["H_j"], round_log["log_P_star_j"])):
                                writer.writerow([sample_idx, round_idx, depth_idx, h, lp])
            logger.info(f"Entropy log saved to {entropy_save_path}")


if __name__ == "__main__":
    main()
