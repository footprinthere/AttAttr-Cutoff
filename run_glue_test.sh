# Arguments
# $1: task name
# $2: GPU number
# $3: attattr version suffix
# $4: training step in which model checkpoint was saved

export GLUE_DIR=/home/jovyan/work/datasets
export TASK_NAME=$1
export SUFFIX=$3
export CKPT_STEP=$4
export BATCH_SIZE=16

CUDA_VISIBLE_DEVICES=$2 python run_glue.py \
  --model_name_or_path roberta-base \
  --saved_dir results/${TASK_NAME}-cutoff-attattr${SUFFIX}/checkpoint-${CKPT_STEP} \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --task_name $TASK_NAME \
  --do_predict \
  --aug_type token_cutoff \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --output_dir results/${TASK_NAME}-cutoff-attattr${SUFFIX}-test
  # --overwrite_output_dir
