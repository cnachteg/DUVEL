# DUVEL [![DOI](https://zenodo.org/badge/679553511.svg)](https://zenodo.org/doi/10.5281/zenodo.10057604)
DUVEL stands for **D**etection of **U**nlimited **V**ariant **E**nsemble in **L**iterature.

## Construction of DUVEL
References from OLIDA (https://olida.ibsquare.be) were selected and annotated with Pubtator with the genes and variants entities. Papers were further filtered to study those containing digenic variant combinations (i.e., variant combinations involving two genes). The candidates were limited to texts containing at most 256 tokens, as well as containing different genes and variants for each candidate. Scripts to create the unlabelled data sets can be found in the ``scripts/construction`` folder.

Annotation was done through the ALAMBIC (https://github.com/Trusted-AI-Labs/ALAMBIC) platform, within an active learning framework with the Margin selection strategy and with an active batch size of 500 samples. The model was a PubMedBERT model, trained for 3 epochs and with a learning rate of 2e-5.

## Fine-tuning experiments
Preliminary experiments were conducted with different biomedical large language models, with hyperparameter fine-tuning. Notebooks experiments can be found in the ``scripts/fine_tuning`` folder.

## Data availibility
Csv files of the data can be found in the ``data`` folder, corresponding to the train/validation/test splits used for the fine-tuning in the experiments of the article.

The train/validation/test splits of the data are also available on Huggingface (https://huggingface.co/datasets/cnachteg/DUVEL) and can be used with the following code :

```python
from datasets import load_dataset
dataset = load_dataset("cnachteg/DUVEL")
```

## Cite us
TBA
