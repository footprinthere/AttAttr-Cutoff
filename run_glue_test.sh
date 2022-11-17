# Arguments
# $1: task name
# $2: number of GPU to use
# $3: suffix
# $4: batch size 

export GLUE_DIR=$WORK/datasets
export TASK_NAME=$1
export NUM_GPU=$2
export SUFFIX=$3
export CKPT_STEP=$4
export BATCH_SIZE=16
export CUTOFF_TYPE="token"

CUDA_VISIBLE_DEVICES=$NUM_GPU \
python run_glue.py \
  --model_name_or_path roberta-base \
  --saved_dir results/${TASK_NAME}-${CUTOFF_TYPE}-cutoff-attr-cache+/checkpoint-${CKPT_STEP} \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --task_name $TASK_NAME \
  --do_predict \
  --aug_type ${CUTOFF_TYPE}_cutoff \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --output_dir results/${TASK_NAME}-${CUTOFF_TYPE}-cutoff-attr-${SUFFIX}-test
  # --overwrite_output_dir
