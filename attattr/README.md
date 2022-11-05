# package `attattr`

## Usage
```python
# import
from attattr import AttrScoreGenerator, ModelInput
```
* 구체적인 사용 방법은 `usage.py` 참조

## Description
* 기존의 AttAttr repo에 구현되어 있던 `generate.attrscore.py`의 기능을, Cutoff와 결합하기 쉽도록 패키지로 재구현

* 기존의 변형된 BERT 모델 구현을 모방해 RoBERTa 모델 재구현
    - Cutoff repo에 구현되어 있는 `transformers`를 베이스로 필요한 기능 추가
    - 임의로 입력한 attention matrix에 대한 model output의 gradient를 추출하는 부분 등

* raw input을 받는 대신 Cutoff 부분에서 처리된 상태의 tokenized input을 전달받도록 구현함

* 기존의 AttAttr repo 구현은 `attattr_orig` 디렉토리에 포함되어 있음
