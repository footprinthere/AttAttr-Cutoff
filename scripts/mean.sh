# Arguments
# $1: task name
# $2: GPU number

export GLUE_DIR=/home/jovyan/work/datasets
export TASK_NAME=$1
export SUFFIX="+mean"
export BATCH_SIZE=16

export PYTHONPAHT=`pwd`

CUDA_VISIBLE_DEVICES=$2 python run_glue.py \
  --model_name_or_path roberta-base \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_aug \
  --aug_type token_cutoff \
  --aug_cutoff_ratio 0.1 \
  --min_cutoff_length 1 \
  --cutoff_except_special_tokens \
  --attr_layer_strategy mean \
  --attr_mean_of_last_layers 2 \
  --aug_ce_loss 1.0 \
  --aug_js_loss 1.0 \
  --learning_rate 5e-6 \
  --num_train_epochs 10.0 \
  --logging_steps 500 \
  --save_steps 500 \
  --save_total_limits 5 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --output_dir results/${TASK_NAME}-cutoff-attattr${SUFFIX}
