# Cutoff_with_Attribution
2022-2 창의적 통합 설계 1 LDI LAB

### Activating virtual environment
```shell
$ conda activate cutoff   # activate
$ conda deactivate
```

### Training
```shell
$ ./run_glue {dataset name} {GPU number} {train batch size} {cutoff type}
# ./run_glue CoLA 0 16 token_cutoff
# cutoff type: span_cutoff, token_cutoff, dim_cutoff
```

## References
**Cutoff** [github](https://github.com/dinghanshen/Cutoff)

**Attention Attribution** [github](https://github.com/YRdddream/attattr)

**Transformer Explainability** [github](https://github.com/hila-chefer/Transformer-Explainability) / [Colab](https://colab.research.google.com/github/hila-chefer/Transformer-Explainability/blob/main/BERT_explainability.ipynb)
