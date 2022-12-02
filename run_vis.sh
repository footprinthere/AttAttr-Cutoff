#Arguments
# $1: task name
# $2: GPU number
# $3: layer strategy

export DATA_DIR=/home/jovyan/work/datasets
export TASK_NAME=$1
export LAYER_STRATEGY=$3

CUDA_VISIBLE_DEVICES=$2 python attrscore_visualize.py \
--model_name_or_path roberta-base \
--task_name $TASK_NAME \
--data_dir $GLUE_DIR/$TASK_NAME \
--attr_layer_strategy $LAYER_STRATEGY \
--attr_mean_of_last_layers 2 \
--output_dir visualize_results/${TASK_NAME}-visualize