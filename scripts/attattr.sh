# special token 배제 옵션과 min cutoff length 옵션 모두 제거

# Arguments
# $1: task name
# $2: GPU number

export GLUE_DIR=/home/jovyan/work/datasets
export TASK_NAME=$1
export SUFFIX=""
export BATCH_SIZE=16

export PYTHONPAHT=`pwd`

if [ "$3" = "weak" ]; then
  export CUTOFF_RATIO=0.05
else
  export CUTOFF_RATIO=0.1
fi

CUDA_VISIBLE_DEVICES=$2 python run_glue.py \
  --model_name_or_path roberta-base \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_aug \
  --aug_type token_cutoff \
  --aug_cutoff_ratio $CUTOFF_RATIO \
  --aug_ce_loss 1.0 \
  --aug_js_loss 1.0 \
  --learning_rate 5e-6 \
  --num_train_epochs 10.0 \
  --logging_steps 500 \
  --save_steps 500 \
  --save_total_limits 5 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --output_dir results/${TASK_NAME}-${3}cutoff-attattr${SUFFIX}
