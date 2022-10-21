PYTHONPATH=`pwd` python examples/generate_attrscore.py \
    --task_name mnli \
    --data_dir model_and_data/mnli_data \
    --bert_model bert-base-cased \
    --batch_size 8 \
    --num_batch 4 \
    --model_file model_and_data/model.mnli.bin \
    --example_index 10 \
    --get_att_attr \
    --output_dir outputs
