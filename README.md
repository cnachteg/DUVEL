# DUVEL
DUVEL stands for **D**etection of **U**nlimited **V**ariant **E**nsemble in **L**iterature.

## Construction of DUVEL
References from OLIDA (https://olida.ibsquare.be) were selected to annotate with Pubtator and create candidates sentences to annotate. Papers were further filtered to study those containing digenic variant combinations (i.e., variant combinations involving two genes). The candidates were limited to text containing at most 256 tokens, as well as containing different genes and variants for each candidate. Scripts to create the unlabelled data sets can be found in the ``scripts/construction`` folder.

Annotation was done through the ALAMBIC (https://github.com/Trusted-AI-Labs/ALAMBIC) platform, within an active learning framework.

## Fine-tuning experiments
Preliminary experiments were conducted with different biomedical large language models, with hyperparameters fine-tuning. Notebooks experiments can be found in the ``scripts/fine_tuning`` folder.

## Data availibility
Csv files of the data can be found in the ``data`` folder, both the complete data set with both labelled and unlabelled data (``all.csv``), as well as the train/validation/test splits used for the fine-tuning in the experiments of the article.

The train/validation/test splits of the data are also available on Huggingface (https://huggingface.co/datasets/cnachteg/DUVEL) and can be used with the following code :

```python
from datasets import load_dataset
dataset = load_dataset("cnachteg/DUVEL")
```

## Cite us
TBA