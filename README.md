This repository contains scripts of coarse-to-fine evaluation for large language models, as detailed in the paper `Towards Coarse-to-Fine Evaluation of Inference Efficiency for Large Language Models`.


# Batch Inference Scenarios

Utilize the following scripts to run batch inference scenarios. Parameters such as `data_path`, `model_name_or_path`, and `backend` can be adjusted to suit your specific needs:

```bash
python batching_inference.py --data_path data/short2short.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --backend vllm
python batching_inference.py --data_path data/short2long.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --backend mii --eval_batch_size 50
python batching_inference.py --data_path data/long2short.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --backend transformers --remove_row_delimiter --eval_batch_size 4
python batching_inference.py --data_path data/short-16k.json --model_name_or_path lmsys/vicuna-7b-v1.5-16k --backend vllm --ignore_eos --max_new_tokens 16000
```

### Notes:
- The `remove_row_delimiter` option helps avoid unintended new lines (`\n`) that some frameworks might generate, ensuring consistent text output length.
- The `ignore_eos` option can be used to prevent the inclusion of End of Sentence tokens in outputs.

# Serving Inference Scenarios

## vLLM Server

Launch a `vLLM` server with the following command:

```bash
python -m vllm.entrypoints.api_server --port 8000 --model meta-llama/Llama-2-7b-chat-hf --dtype bfloat16
```

Evaluate the server with these commands:

```bash
python serving_inference.py --backend vllm --api_url http://127.0.0.1:8000/generate --data_path data/short2short.json --request_rate 2
python serving_inference.py --backend vllm --api_url http://127.0.0.1:8000/generate --data_path data/short2long.json --request_rate 2
python serving_inference.py --backend vllm --api_url http://127.0.0.1:8000/generate --data_path data/long2short.json --request_rate 2
```

The `data_path`, `api_url`, and `request_rate` in the above command line commands can be modified as needed.

## DeepSpeed-MII Server

Launch a `DeepSpeed-MII` server using:

```bash
python launch_mii.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf --max_length 4096 --gpu_id 0
```

Evaluate the server with these commands:

```bash
python serving_inference.py --backend mii --api_url http://127.0.0.1:18000/mii/Llama-2-13b-chat-hf --data_path data/short2short.json --request_rate 2
python serving_inference.py --backend mii --api_url http://127.0.0.1:18000/mii/Llama-2-13b-chat-hf --data_path data/short2long.json --request_rate 2
python serving_inference.py --backend mii --api_url http://127.0.0.1:18000/mii/Llama-2-13b-chat-hf --data_path data/long2short.json --request_rate 2
```

## Modifiable Parameters:
- `data_path`, `api_url`, and `request_rate` can be adjusted according to your scenario requirements to achieve optimal results.

# Fine-grained Modular Evaluation

Here, we provide scripts in `fine-grained` directory that allow you to obtain fine-grained performance metrics for `transformers` and `vllm` models using Nsight Compute CLI. Below you'll find the instructions on how to set up your environment and run the scripts.

## Prerequisites

Before you begin, make sure you have the following installed on your machine:

- [PyTorch](https://pytorch.org/) - An open-source machine learning library.
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) - A performance analysis tool that provides detailed insights into CUDA-based applications.

## Environment Setup

To use the Python-based interface support for Nsight Compute, you need to set the `PYTHONPATH` environment variable. Replace `/PATH/TO/YOUR/NSIGHT/COMPUTE` with the actual path to your Nsight Compute installation:

```bash
export PYTHONPATH=/PATH/TO/YOUR/NSIGHT/COMPUTE/extras/python:$PYTHONPATH
```

## Running the Scripts

The scripts are organized into two directories:

- `fine-grained/transformers`
- `fine-grained/vllm`

To get the fine-grained performance results, navigate to one of the above directories and run the following commands. This will execute the scripts and save the results in the `ans` directory.

```bash
bash run.sh
bash ncu.sh
bash profile.sh
```

## Customizing Parameters

You can customize various hyper-parameters to suit your needs. These include:

- `CUDA_VISIBLE_DEVICES` - Specifies which GPUs to use.
- `prompt_len` - Defines the length of the prompt.
- `batch_size` - Sets the number of samples processed for model.
- `ncu_path` - The path to the report of the Nsight Compute CLI executable.
- `events_path` - The path where profiling events are stored.

To change these parameters, you will need to edit the corresponding values in the script files or pass them as environment variables before running the scripts.

# Libraries Version

| Lib. | Version |
|------|---------|
| CUDA | 12.1    |
| Nsight Compute | 2023.1.1 |
| Transformers  | 4.36.2 |
| vLLM          | 0.2.7  |
| DeepSpeed-MII | 0.1.3  |
| TensorRT-LLM  | [link](https://github.com/NVIDIA/TensorRT-LLM/tree/0ab9d17a59c284d2de36889832fe9fc7c8697604) |
| llama.cpp     | [link](https://github.com/ggerganov/llama.cpp/tree/122ed4840cc6d209df6043e027f9f8a03aee01da) |

