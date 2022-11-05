import os
from attrscore import AttrScoreGenerator


def test_main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    generator = AttrScoreGenerator(
        model_name='roberta-base',
        task_name='mnli',
        model_file='/home/jovyan/work/checkpoint/MNLI/checkpoint_token/pytorch_model.bin'
    )

    inputs = generator.process_input(
        data_dir='/home/jovyan/work/seongtae/AttAttr-Cutoff/attattr/model_and_data/mnli_data',
        example_index=10,
    )

    attr = generator.genereate_attrscore(inputs)

    generator.dump_attr(attr, output_dir='outputs')


if __name__ == '__main__':
    test_main()
