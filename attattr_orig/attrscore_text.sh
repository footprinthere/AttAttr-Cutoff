export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=$1

python examples/attscore.py \
    --model_name bert-base-cased \
    --model_file model_and_data/model.mnli.bin \
    --task_name mnli \
    --data_dir model_and_data/mnli_data \
    --example_index 10 \
    --batch_size 8 \
    --num_batches 4 \
    --output_dir outputs
