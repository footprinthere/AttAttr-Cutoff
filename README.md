# Cutoff_with_Attribution
2022-2 창의적 통합 설계 1 LDI LAB

### Activating virtual environment
```shell
$ conda activate cutoff   # activate
$ conda deactivate
```

### Training
```shell
$ ./run_glue.sh {dataset name} {GPU number}
# ./run_glue.sh CoLA 0
```

### Testing
```shell
$ ./run_glue_test.sh {dataset name} {GPU number} {checkpoint step}
# ./run_glue_test.sh CoLA 0 100
# if your model checkpoint file is saved under the directory "checkpoint-100"
```

### Cutoff types
```
span_cutoff, token_cutoff, dim_cutoff
```

## References
**Cutoff** [github](https://github.com/dinghanshen/Cutoff)

**Attention Attribution** [github](https://github.com/YRdddream/attattr)

**Transformer Explainability** [github](https://github.com/hila-chefer/Transformer-Explainability) / [Colab](https://colab.research.google.com/github/hila-chefer/Transformer-Explainability/blob/main/BERT_explainability.ipynb)
