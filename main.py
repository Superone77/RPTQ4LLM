import os
import sys

import random
import numpy as np
from models.llama import LlamaClass
import torch
import time
from datautils import get_loaders
from lm_eval import tasks, evaluator
from quantize.llama_reorder_quantize import llama_reorder_quantize
import datetime
from models.int_llama_layer import QuantLlamaAttention
from pprint import pprint
import torch.nn as nn
from tqdm import tqdm

net_choices = [
    "llama-7b",
    "llama-13b",
    "llama3-8b",
    "llama3-70b",
]

# tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq


@torch.no_grad()
def evaluate(lm, args):
    for name, m in lm.model.named_modules():
        if isinstance(m, QuantLlamaAttention):
            m.name = name

    results = {}
    lm.model.model = lm.model.model.to(lm.device)

    if args.eval_ppl:
        for dataset in ["wikitext2", "ptb", "c4"]:
            cache_testloader = f"/tmp/{dataset}_testloader_llama_all.cache"
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                    cache_dir=args.cache_dir,
                )
                torch.save(testloader, cache_testloader)
            # print(dataset)
            if "c4" == dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []

            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(
                    lm.device
                )
                outputs = lm.model.model(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            print(dataset, ppl.item())
            lm.model.config.use_cache = use_cache
            # pprint(args.model)
            results[dataset] = ppl.item()
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        pprint(results)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, choices=net_choices)
    parser.add_argument(
        "--cache_dir", default="./data", type=str, help="model cache directory"
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="mix",
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ema_minmax",
        choices=["minmax", "ema_minmax", "mse", "layer_mse"],
    )

    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output_path", default="./output")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--disable_w_quant", action="store_true")
    parser.add_argument("--disable_a_quant", action="store_true")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to local LLaMA model directory",
    )
    parser.add_argument("--R1_clusters", type=int, default=32)
    parser.add_argument("--R2_clusters", type=int, default=4)
    parser.add_argument("--R3_clusters", type=int, default=4)
    parser.add_argument("--R4_clusters", type=int, default=32)
    parser.add_argument("--R5_clusters", type=int, default=32)
    parser.add_argument("--reorder", type=str, default="12345", help="like 12345 or 1")
    parser.add_argument(
        "--w_quantizer", type=str, default="gptq", choices=["gptq", "normal"]
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--a_dynamic", action="store_true")
    parser.add_argument("--eval_base_ppl", action="store_true")
    parser.add_argument("--act_dist_plot", action="store_true")
    parser.add_argument("--only_quant_kv", action="store_true")
    parser.add_argument(
        "--pack_weight",
        action="store_true",
        help="enable this to reduce memory consumption",
    )
    parser.add_argument(
        "--hidden_layer_num",
        type=int,
        default=None,
        help="Number of LLaMA hidden layers to use for quick debugging",
    )

    args = parser.parse_args()
    args.batch_size = 1  # BS=1 is used for zeroShot tasks!
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if "llama3" in args.net:
        size = args.net.split('-')[1]
        if args.model_path is not None:
            args.model = args.model_path
        else:
            args.model = f"meta-llama/Meta-Llama-3-{size.upper()}"
        lm = LlamaClass(args)
        lm.model.eval()
    elif "llama" in args.net:
        size = args.net.split('-')[1]
        if args.model_path is not None:
            args.model = args.model_path
        else:
            args.model = f"meta-llama/Llama-2-{size}-hf"
        lm = LlamaClass(args)
        lm.model.eval()
    else:
        raise NotImplementedError

    print("=== start quantization ===")
    if args.load:
        print("Loading checkpoint from {}...".format(args.load))
        lm.model.load_state_dict(torch.load(args.load))

    tick = time.time()

    cache_dataloader = (
        f"/tmp/dataloader_llama_{args.calib_dataset}_{args.nsamples}.cache"
    )
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        print(f"load calibration from {cache_dataloader}")
    else:
        dataloader, testloader = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
            cache_dir=args.cache_dir,
        )
        torch.save(dataloader, cache_dataloader)
    lm.model.eval()

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": False,
        "metric": "minmax",
    }
    args.act_quant_params = {
        "n_bits": 16 if args.only_quant_kv else args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.q_quant_params = {
        "n_bits": 16 if args.only_quant_kv else args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.layer_norm_out_quant_params = {
        "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.p_quant_params = {
        "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
        "metric": "fix0to1",
    }
    n_clusters = {
        "R1": args.R1_clusters,
        "R2": args.R2_clusters,
        "R3": args.R3_clusters,
        "R4": args.R4_clusters,
        "R5": args.R5_clusters,
    }

    llama_reorder_quantize(
        lm,
        args,
        dataloader,
        n_clusters,
        args.reorder,
    )

    for layer in lm.model.model.layers:
        if hasattr(layer, "set_quant_state"):
            layer.set_quant_state(
                not args.disable_w_quant, not args.disable_a_quant
            )

    print(time.time() - tick)

    results = evaluate(lm, args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(
        f"{args.output_path}/{args.net}.txt",
        "a+",
    ) as f:
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"{' '.join(sys.argv)} {formatted_time} \n {args} \n w{args.wbits}a{args.abits} {results}\n\n"
        )


if __name__ == "__main__":
    print(sys.argv)
    main()
