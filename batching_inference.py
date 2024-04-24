import os
import time
import json
import argparse
import logging
import random
import queue
import asyncio
from typing import Dict, Tuple, List, Any, Union, DefaultDict
import hashlib
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetName


from tqdm import tqdm
import vllm
import torch
import transformers
import mii
from deepspeed.inference.config import DtypeEnum
from mii.batching.ragged_batching import *
from mii.batching.data_classes import Response
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList


class StopLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, stop='\n\n\n\n'):
        self.stop_list = torch.tensor([13, 13, 13, 13]).cuda() # tokenizer.encode(stop, add_special_tokens=False, return_tensors="pt").cuda()
        self.inf = torch.tensor(1e5).cuda()
        self.stop_length = len(self.stop_list)
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.all(input_ids[:, -self.stop_length:] == self.stop_list, dim=1)
        scores[:, self.eos_token_id] = torch.where(mask, self.inf, scores[:, self.eos_token_id])
        return scores


@torch.no_grad()
def main(args):
    random.seed(42)

    logging.info("loading data and model...")
    if args.data_path is None:
        raise ValueError
    with open(args.data_path, 'r') as f:
        eval_data = json.load(f)
    dataset_name = os.path.split(args.data_path)[-1]

    if 'Llama-2' in args.model_name_or_path:
        prompts = [example['prompt'] for example in eval_data]
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        print(prompts[0])
    elif 'vicuna' in args.model_name_or_path:
        prompts = [example['prompt'] for example in eval_data]
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        # tokenizer.pad_token = tokenizer.unk_token
        # tokenizer.pad_token_id = tokenizer.unk_token_id
        print(prompts[0])
    else:
        raise NotImplementedError

    if args.backend == 'vllm':
        model = vllm.LLM(
            model=args.model_name_or_path,
            dtype=torch.bfloat16,
            tokenizer=args.model_name_or_path,
            tensor_parallel_size=1,
        )
        sampling_kwargs = dict(
            temperature=0,  # greedy decoding
            max_tokens=args.max_new_tokens,
        )
        if args.remove_row_delimiter:
            sampling_kwargs['stop'] = '\n\n\n\n'
        if args.ignore_eos:
            sampling_kwargs['ignore_eos'] = True
        sampling_params = vllm.SamplingParams(**sampling_kwargs)

        start_time = time.time()
        out = model.generate(prompts, sampling_params=sampling_params)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'total_time = {total_time}')
        print(f'sequence num = {len(prompts)}')
        total_token_num = 0
        outputs = []
        for it in out:
            total_token_num += len(it.outputs[0].token_ids)
            outputs.append(
                it.outputs[0].text
            )
        print(f'generated tokens num = {total_token_num}')
    elif args.backend == 'mii':
        max_length = 4096 if 'llama2' in args.model_name_or_path else 16383
        # pipe = mii.pipeline(args.model_name_or_path, max_length=max_length, torch_dist_port=11454)
        total_time = 0
        total_token_num = 0
        outputs = []
        pipe_kwargs = {
            'do_sample': False,
            'max_new_tokens': args.max_new_tokens,
        }
        if args.ignore_eos:
            pipe_kwargs['ignore_eos'] = True
            # pipe_kwargs['stop'] = -1
        for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
            pipe = mii.pipeline(args.model_name_or_path, max_length=max_length, torch_dist_port=11454)
            start_time = time.time()
            out = pipe(prompts[i:i + args.eval_batch_size], **pipe_kwargs)
            end_time = time.time()
            pipe.destroy()
            del pipe
            total_time += end_time - start_time
            # print(len(out))
            for it in out:
                total_token_num += it.generated_length
                outputs.append(it.generated_text)
            # break
        print(f'total_time = {total_time}')
        print(f'sequence num = {len(prompts)}')
        print(f'generated tokens num = {total_token_num}')
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).eval().cuda()
        total_time = 0
        total_token_num = 0
        outputs = []
        generation_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False
        )
        if args.remove_row_delimiter:
            processors = LogitsProcessorList()
            processors.append(StopLogitsProcessor(tokenizer, '\n\n\n\n'))
            generation_kwargs['logits_processor'] = processors
        if args.ignore_eos:
            generation_kwargs['eos_token_id'] = -1
        for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
            start_time = time.time()
            inputs = tokenizer(prompts[i:i + args.eval_batch_size], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
            input_ids = inputs['input_ids'].cuda()
            output = model.generate(input_ids, **generation_kwargs)
            out_str = tokenizer.batch_decode(output[:, input_ids.size(1):], skip_special_tokens=True)
            end_time = time.time()
            total_time += end_time - start_time
            token_num = int((output[:, input_ids.size(1):] != tokenizer.pad_token_id).sum().cpu())
            total_token_num += token_num
            for prompt, o_str in zip(prompts[i:i + args.eval_batch_size], out_str):
                outputs.append(o_str)
        print(f'total_time = {total_time}')
        print(f'sequence num = {len(prompts)}')
        print(f'generated tokens num = {total_token_num}')

    handle = nvmlDeviceGetHandleByIndex(int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')))
    device_name = nvmlDeviceGetName(handle)
    result = {
        'backend': args.backend,
        'dataset': dataset_name,
        'device': device_name,
        'total_time': total_time,
        'sequence_num': len(prompts),
        'generated_tokens_num': total_token_num,
        'result': [],
    }
    for prompt, output in zip(prompts, outputs):
        result['result'].append({
            'prompt': prompt,
            'output': output,
        })
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    output_file = f'output/{hashlib.md5(json_str.encode("utf-8")).hexdigest()}.json'
    print(f'The result will be saved in {output_file}')
    with open(output_file, 'w') as f:
        f.write(json_str)


if __name__ == '__main__':
    assert len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')) == 1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="If specified, we will load the data to generate the predictions.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=8, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm", "mii"]
    )
    parser.add_argument(
        "--remove_row_delimiter",
        action="store_true",
        help="If given, we will remove row delimiter repeated four times.",
    )
    parser.add_argument(
        "--ignore_eos",
        action="store_true",
        help="If given, we will continue generating tokens after the EOS token is generated.",
    )
    args = parser.parse_args()

    main(args)
