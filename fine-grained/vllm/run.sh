export CUDA_VISIBLE_DEVICES=3
for prompt_len in 32 128 512 1024 2048 4090; do
	echo $prompt_len
for batch_size in 8 ; do
    python pickle_vllm.py  $batch_size $prompt_len
done
done

# for prompt_len in 1024; do
# 	echo $prompt_len
# for batch_size in 1 2 4 8 16 ; do
#     python test_vllm.py  $batch_size $prompt_len 1
# done
# done