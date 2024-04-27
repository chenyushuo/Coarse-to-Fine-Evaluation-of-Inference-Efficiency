for prompt_len in 32 128 512 1024 2048; do
for batch_size in 8 ; do
    python profile_transformers.py --ncu_path /media/profile/transformers/test_${batch_size}_${prompt_len}.ncu-rep --events_path ./data/tests_${batch_size}_${prompt_len}.pkl > transformers_${batch_size}_${prompt_len}
done
done

# for prompt_len in 1024; do
# for batch_size in 1 2 4 8 16 ; do
#     python transformers_profile.py --ncu_path /media/profile/transformers/test_${batch_size}_${prompt_len}_3090.ncu-rep --events_path ../transformers_time/tests_${batch_size}_${prompt_len}_3090.pkl > transformers_${batch_size}_${prompt_len}_3090
# done
# done