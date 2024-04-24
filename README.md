The scripts of `Towards Coarse-to-Fine Evaluation of Inference Efficiency for Large Language Models`.


# Batch Inference Scenarios

```bash
python batching_inference.py --data_path data/short2short.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --backend vllm
python batching_inference.py --data_path data/short2long.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --backend mii --eval_batch_size 50
python batching_inference.py --data_path data/long2short.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --backend transformers --remove_row_delimiter --eval_batch_size 4
python batching_inference.py --data_path data/short-16k.json --model_name_or_path lmsys/vicuna-7b-v1.5-16k --backend vllm --ignore_eos --max_new_tokens 16000
```

# Serving Inference Scenarios

## vLLM

Here is the script to launch `vLLM` server:

```bash
python -m vllm.entrypoints.api_server --port 8000 --model meta-llama/Llama-2-7b-chat-hf --dtype bfloat16
```

The script to evaluate serving scenario of `vLLM`:

```bash
python serving_inference.py --backend vllm --api_url http://127.0.0.1:8000/generate --data_path data/short2short.json --request_rate 2
python serving_inference.py --backend vllm --api_url http://127.0.0.1:8000/generate --data_path data/short2long.json --request_rate 2
python serving_inference.py --backend vllm --api_url http://127.0.0.1:8000/generate --data_path data/long2short.json --request_rate 2
```

## DeepSpeed-MII

The script to launch `DeepSpeed-MII` server:

```bash
python launch_mii.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf --max_length 4096 --gpu_id 0
```

The script to evaluate serving scenario of `DeepSpeed-MII`:

```bash
python serving_inference.py --backend mii --api_url http://127.0.0.1:18000/mii/Llama-2-13b-chat-hf --data_path data/short2short.json --request_rate 2
python serving_inference.py --backend mii --api_url http://127.0.0.1:18000/mii/Llama-2-13b-chat-hf --data_path data/short2long.json --request_rate 2
python serving_inference.py --backend mii --api_url http://127.0.0.1:18000/mii/Llama-2-13b-chat-hf --data_path data/long2short.json --request_rate 2
```

# Fine-grained Modular Evaluation


# Libraries Version

| Lib. | Version |
|------|---------|
| Transformers  | 4.36.2 |
| vLLM          | 0.2.7  |
| DeepSpeed-MII | 0.1.3  |
| TensorRT-LLM  | [link](https://github.com/NVIDIA/TensorRT-LLM/tree/0ab9d17a59c284d2de36889832fe9fc7c8697604) |
| llama.cpp     | [link](https://github.com/ggerganov/llama.cpp/tree/122ed4840cc6d209df6043e027f9f8a03aee01da) |

