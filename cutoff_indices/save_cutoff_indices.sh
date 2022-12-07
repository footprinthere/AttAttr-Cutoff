# $1: task name (lower case)
# $2: GPU
# $3: layer strategy

export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=$2
export SAVE_DIR="./cutoff_indices/$1"
export SAVED_EXAMPLES=300

python cutoff_indices/save_cutoff_indices.py \
    --task_name $1 \
    --save_dir $SAVE_DIR \
    --save_period 500 \
    --attr_layer_strategy $3
    # --saved_npy_10 "$SAVE_DIR/temp/$1_indices_${SAVED_EXAMPLES}_10.npy" \
    # --saved_npy_05 "$SAVE_DIR/temp/$1_indices_${SAVED_EXAMPLES}_05.npy" \
    # --skip_first_n_examples $SAVED_EXAMPLES
