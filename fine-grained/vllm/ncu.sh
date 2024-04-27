export CUDA_VISIBLE_DEVICES=3
for prompt_len in 32 128 512 1024 2048 4090; do
for batch_size in 8 ; do
    echo start_${batch_size}_${prompt_len} >> vllm.log
    ncu --config-file off --export /media/profile/vllm/test_${batch_size}_${prompt_len} --force-overwrite --section-folder ../sections --section MemoryWorkloadAnalysis_Chart --rule Memory --replay-mode application python ncu_vllm.py  $batch_size $prompt_len 1
    echo finish_${batch_size}_${prompt_len} >> vllm.log
done
done

# for batch_size in 1 2 4 8 16; do
# for prompt_len in 1024; do
#     echo start_${batch_size}_${prompt_len} >> vllm.log
#     ncu --config-file off --export /media/profile/vllm/test_${batch_size}_${prompt_len}_3090 --force-overwrite --section-folder /media/profile/vllm/sections --section MemoryWorkloadAnalysis_Chart --rule Memory --replay-mode application python test_vllm.py  $batch_size $prompt_len 1 
#     echo finish_${batch_size}_${prompt_len} >> vllm.log
# done
# done