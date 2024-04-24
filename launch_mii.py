import argparse
import mii
import os
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum number of model input."
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    client = mii.serve(
        args.model_name_or_path,
        max_length=args.max_length,
        deployment_name=os.path.split(args.model_name_or_path)[-1],
        enable_restful_api=True,
        restful_api_port=18000,
        hostfile='./deepspeed-hostfile',
        device_map={'localhost': [[args.gpu_id]]},
        torch_dist_port=11451,
        port_number=50150
    )
