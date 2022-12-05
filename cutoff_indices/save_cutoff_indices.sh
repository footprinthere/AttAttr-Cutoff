# $1: task name (lower case)
# $2: GPU
# $3: layer strategy

export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=$2

python cutoff_indices/save_cutoff_indices.py \
    --task_name $1 \
    --save_dir "./cutoff_indices" \
    --save_period 500 \
    --attr_layer_strategy $3
