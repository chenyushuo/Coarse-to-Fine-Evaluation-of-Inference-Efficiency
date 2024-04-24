import os
import argparse
import asyncio
import json
import random
import time
import hashlib

from tqdm import tqdm
import httpx
import aiohttp
import transformers
import numpy as np


# query_start_time = None
async def vllm_inference(url, prompt, idx, outputs, tqdm_info, **kwargs):
    headers = {"User-Agent": "Test Client"}
    data = {
        "prompt": prompt,
        "temperature": 0.0,
    }
    data.update(kwargs)
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        response = await session.post(url, headers=headers, json=data, timeout=None)
        data = await response.json()
        output = data['text'][0][len(prompt):]
        tqdm_info['bar'].update(1)
    end_time = time.time()
    request_time = end_time - start_time
    outputs[idx] = {
        'prompt': prompt,
        'output': output,
        'start_time': start_time,
        'end_time': end_time,
        'latency': request_time,
    }


async def mii_inference(url, prompt, idx, outputs, tqdm_info, **kwargs):
    headers = {"Content-Type": "application/json"}
    data = {
        "prompts": [prompt],
        "do_sample": False,
    }
    data.update(kwargs)
    start_time = time.time()
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(url, headers=headers, json=data, timeout=None)
                data = await response.json()
                output = data[0]['generated_text']
                tqdm_info['bar'].update(1)
            break
        except Exception as e:
            continue
    end_time = time.time()
    request_time = end_time - start_time
    outputs[idx] = {
        'prompt': prompt,
        'output': output,
        'start_time': start_time,
        'end_time': end_time,
        'latency': request_time,
    }


async def llama_cpp_inference(url, prompt, idx, outputs, tqdm_info, **kwargs):
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "temperature": 0,
    }
    data.update(kwargs)
    start_time = time.time()
    # while True:
    #     try:
    #         async with aiohttp.ClientSession() as session:
    #             response = await session.post(url, headers=headers, json=data, timeout=None)
    #             result = await response.json()
    #             output = result['content']
    #             tqdm_info['bar'].update(1)
    #         break
    #     except Exception as e:
    #         continue
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                response = await session.post(url, headers=headers, json=data, timeout=None)
                result = await response.json()
                output = result['content']
                tqdm_info['bar'].update(1)
                break
            except Exception as e:
                # print(e)
                await asyncio.sleep(0.004)
                continue
    end_time = time.time()
    request_time = end_time - start_time
    # print(data)
    outputs[idx] = {
        'prompt': prompt,
        'output': output,
        'start_time': start_time,
        'end_time': end_time,
        'latency': request_time,
    }


def get_eval_data(args):
    if args.data_path is None:
        raise ValueError
    with open(args.data_path, 'r') as f:
        eval_data = json.load(f)
    if 'Llama-2' in args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    else:
        raise NotImplementedError

    return eval_data, tokenizer


async def benchmark(eval_data, args):
    iter_data_tqdm = tqdm(total=len(eval_data), desc='Send Requests    ')
    async def iter_data(max_tokens_name):
        # start_time = time.time()
        for i, data in enumerate(eval_data):
            kwargs = {max_tokens_name: data['max_tokens']}
            last_time = time.time()
            iter_data_tqdm.update(1)
            yield i, data['prompt'], kwargs
            if args.request_rate == float('inf'):
                continue
            interval = np.random.exponential(1.0 / args.request_rate) - (time.time() - last_time)
            # assert interval > 0, f'{interval}, {(time.time() - last_time)}'
            if interval > 0:
                await asyncio.sleep(interval)
        # print(f'real rate = {1000 / (last_time - start_time)}')
    tasks = []
    outputs = [None] * len(eval_data)
    tqdm_info = {
        'bar': tqdm(total=len(eval_data), desc='Finished Requests'),
        'client_num': args.client_num,
    }
    if args.backend == 'vllm':
        reqeust_func = vllm_inference
        max_tokens_name = 'max_tokens'
        # kwargs = {
        #     "max_tokens": args.max_new_tokens,
        # }
    elif args.backend == 'mii':
        reqeust_func = mii_inference
        max_tokens_name = 'max_new_tokens'
        # kwargs = {
        #     "max_length": args.max_new_tokens,
        # }
    elif args.backend == 'llama.cpp':
        reqeust_func = llama_cpp_inference
        max_tokens_name = 'n_predict'
    else:
        raise NotImplementedError

    async for idx, prompt, kwargs in iter_data(max_tokens_name):
        task = asyncio.create_task(
            reqeust_func(args.api_url, prompt, idx, outputs, tqdm_info, **kwargs)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)
    return outputs


def main(args):
    random.seed(args.seed)
    eval_data, tokenizer = get_eval_data(args)

    detailed_log = {
        'args': vars(args),
        'version': 'v4.1',
        'outputs': [],
    }
    profile_log = {
        'total_time': [],
        'sequence_throughput': [],
        'real_request_rate': [],
        'generated_tokens': [],
        'avg_latency': [],
        'avg_latency_per_token': [],
        'avg_latency_per_output_token': [],
    }
    for round in range(args.repeat_count):
        print(f'Round {round}:')
        outputs = asyncio.run(benchmark(eval_data, args))
        detailed_log['outputs'].append(outputs)

        total_time = outputs[-101]['end_time'] - outputs[100]['start_time']
        print(f'Total time: {total_time:.2f} s')
        seq_throughput = (len(eval_data) - 200) / total_time
        print(f'Sequence throughput: {seq_throughput:.2f} requests/s')
        profile_log['total_time'].append(total_time)
        profile_log['sequence_throughput'].append(seq_throughput)
        real_request_rate = (len(eval_data) - 200) / (outputs[-101]['start_time'] - outputs[100]['start_time'])
        print(f'Real Request rate = {real_request_rate:.2f} requests/s')
        profile_log['real_request_rate'].append(real_request_rate)

        middle_slice = slice(100, -100)
        prompts_ids = tokenizer(
            [
                output['prompt']
                for output in outputs
            ],
            add_special_tokens=False,
        )['input_ids']
        outputs_ids = tokenizer(
            [
                output['output']
                for output in outputs
            ],
            add_special_tokens=False,
        )['input_ids']
        prompts_length = list(map(len, prompts_ids))
        outputs_length = list(map(len, outputs_ids))
        total_generated_tokens = sum(outputs_length[middle_slice])
        print(f"Total generated tokens: {total_generated_tokens}")
        profile_log['generated_tokens'].append(total_generated_tokens)

        avg_latency = np.mean([output['latency'] for output in outputs[middle_slice]])
        print(f"Average latency: {avg_latency * 1000:.2f} ms")
        profile_log['avg_latency'].append(avg_latency)
        avg_per_token_latency = np.mean([
            output['latency'] / (prompt_len + output_len)
            for prompt_len, output_len, output in zip(
                prompts_length[middle_slice], outputs_length[middle_slice], outputs[middle_slice]
            )
        ])
        print(f"Average latency per token: {avg_per_token_latency * 1000:.2f} ms")
        profile_log['avg_latency_per_token'].append(avg_per_token_latency)
        avg_per_output_token_latency = np.mean([
            output['latency'] / output_len
            for output_len, output in zip(
                outputs_length[middle_slice], outputs[middle_slice]
            )
        ])
        print("Average latency per output token: "
            f"{avg_per_output_token_latency * 1000:.2f} ms")
        profile_log['avg_latency_per_output_token'].append(avg_per_output_token_latency)
        print()

    print('Summary:')
    print(f'Real request rate: {np.mean(profile_log["real_request_rate"]):.2f}'
        f' ± {np.std(profile_log["real_request_rate"]):.2f} requests/s')
    print(f'Total time: {np.mean(profile_log["total_time"]):.2f}'
        f' ± {np.std(profile_log["total_time"]):.2f} s')
    print(f'Sequence throughput: {np.mean(profile_log["sequence_throughput"]):.2f}'
        f' ± {np.std(profile_log["sequence_throughput"]):.2f} requests/s')
    print(f'Total generated tokens: {np.mean(profile_log["generated_tokens"]):.2f}'
        f' ± {np.std(profile_log["generated_tokens"]):.2f}')
    print(f'Average latency: {np.mean(profile_log["avg_latency"]):.2f}'
        f' ± {np.std(profile_log["avg_latency"]):.2f} s')
    print(f'Average latency per token: {np.mean(profile_log["avg_latency_per_token"]) * 1000:.2f}'
        f' ± {np.std(profile_log["avg_latency_per_token"]) * 1000:.2f} ms')
    print(f'Average latency per output token: {np.mean(profile_log["avg_latency_per_output_token"]) * 1000:.2f}'
        f' ± {np.std(profile_log["avg_latency_per_output_token"]) * 1000:.2f} ms')

    md5sum_of_log_meta = hashlib.md5(json.dumps(
        {
            'args': detailed_log['args'],
            'version': detailed_log['version'],
        }
    ).encode('utf-8')).hexdigest()
    output_file = f"output/benchmark/{md5sum_of_log_meta}.json"
    with open(output_file, 'w') as f:
        json.dump(detailed_log, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--repeat_count",
        type=int,
        default=5,
        help="The number of experimental repetitions."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "mii", "llama.cpp"]
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default=None,
        help="If specified, we will use this url to generate.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='/run/user/1000/Llama-2-7b-chat-hf',
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="If specified, we will load the data to generate the predictions.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/result.jsonl",
        help="Path of file to save generated results.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--client_num",
        type=int,
        default=2**32,
        help="Number of client."
    )
    parser.add_argument(
        "--request_rate",
        type=float,
        default=float('inf'),
        help="Number of requests per second."
    )
    args = parser.parse_args()

    main(args)
