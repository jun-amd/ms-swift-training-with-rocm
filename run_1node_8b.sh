pip install 'ms-swift'
pip install pybind11
pip install git+https://github.com/amd-fuweiy/Megatron-LM.git@core_v0.15.0-rocm_fix
sed -i 's/max_version = PkgVersion("2\.8\.0\.post2")/max_version = PkgVersion("2.8.3")/' /opt/venv/lib/python3.10/site-packages/transformer_engine/pytorch/dot_product_attention/utils.py

export MODELSCOPE_CACHE=${PWD}/shared
export MEGATRON_LM_PATH=${PWD}/Megatron-LM

# If you encouter connection (to modelscope) timeout during training, you could download them before training.
# - modelscope download --model Qwen/Qwen3-30B-A3B
# - modelscope download --dataset swift/chinese-c4

output_dir=${PWD}/megatron_output
mkdir -p ${output_dir}
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

seq=32768
mbs=1
gbs=128
tp=1
pp=1
ep=1

log_file=${output_dir}/"1node_swift_qwen3_8b_seq${seq}_tp${tp}_pp${pp}_ep${ep}_mbs${mbs}_gbs${gbs}_${current_time}.log"

NPROC_PER_NODE=8 \
megatron pt \
    --model ${MODELSCOPE_CACHE}/models/Qwen/Qwen3-8B \
    --dataset ${MODELSCOPE_CACHE}/datasets/swift/chinese-c4 \
    --streaming true \
    --tensor_model_parallel_size ${tp} \
    --pipeline_model_parallel_size ${pp} \
    --expert_model_parallel_size ${ep} \
    --load_safetensors true \
    --save_safetensors true \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --split_dataset_ratio 0.01 \
    --micro_batch_size ${mbs} \
    --global_batch_size ${gbs} \
    --packing true \
    --train_iters 2000 \
    --cross_entropy_loss_fusion true \
    --eval_iters 50 \
    --finetune false \
    --lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-7 \
    --save megatron_output/Qwen3-8B \
    --eval_interval 20000 \
    --save_interval 20000 \
    --max_length ${seq} \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --log-throughput \
    --log-interval 1 \
    --use_distributed_optimizer false \
    --overlap_grad_reduce true \
    2>&1 | tee ${log_file}
