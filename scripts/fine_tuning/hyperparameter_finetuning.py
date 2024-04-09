import torch
import random
import wandb
import numpy as np
import evaluate
import gc
import os

from transformers import (
    TrainingArguments, 
    Trainer,
    AutoTokenizer, 
    DefaultDataCollator,
    AutoModelForSequenceClassification,
)
from datasets import (
    load_dataset, 
    Features, 
    ClassLabel, 
    Value,
)

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
os.environ['WANDB_PROJECT']='DUVEL_finetune'
os.environ['TOKENIZERS_PARALLELISM']="true"

num_gpus = torch.cuda.device_count()

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def compute_metrics_fn(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    recall = evaluate.load('recall')
    precision = evaluate.load('precision')
    f1 = evaluate.load('f1')
    wandb.log({
        'recall': recall.compute(predictions=predictions, references=labels)['recall'],
        'precision':precision.compute(predictions=predictions, references=labels)['precision'],
        'f1_score':f1.compute(predictions=predictions, references=labels)['f1']
    })
    return {
        'recall': recall.compute(predictions=predictions, references=labels)['recall'],
        'precision':precision.compute(predictions=predictions, references=labels)['precision'],
        'f1_score':f1.compute(predictions=predictions, references=labels)['f1']
    }

def train(config=None):
    with wandb.init(config=config):
        
        # set sweep configuration
        config = wandb.config
        tokenizer = AutoTokenizer.from_pretrained(config.model_name) # use_fast=True)
        dataset = load_dataset("cnachteg/duvel",use_auth_token=True)
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
        
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        
        def collate_fn(example):
            # we have to do it padding full here and not dynamic because the predict function in
            # strategy does not do the collate_fn
            outputs = tokenizer(
            example["sentence"],
            truncation = True,
            max_length = 256,
            padding="max_length",
            )
            outputs["labels"] = example["label"]
            return outputs
        
        processed_dataset_train = train_dataset.map(
            collate_fn, 
            batched=True,
            remove_columns=['sentence', 'pmcid', 'gene1', 'gene2', 'variant1', 'variant2', 'label']
        )
        processed_dataset_eval = eval_dataset.map(
                collate_fn,
                batched=True,
                remove_columns=['sentence', 'pmcid', 'gene1', 'gene2', 'variant1', 'variant2', 'label']
            )


        # set training arguments
        training_args = TrainingArguments(
            run_name=f'batch-{config.batch_size}_lr-{config.learning_rate}_epochs-{config.epochs}',
            output_dir='DUVEL_finetune',
            report_to='wandb',  # Turn on Weights & Biases logging
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate*num_gpus,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=8,
            save_strategy='no',
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            remove_unused_columns=False,
            seed=42,
            do_predict=True,
            do_train=True,
            do_eval=True
        )


        # define training loop
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            data_collator=DefaultDataCollator(),
            train_dataset=processed_dataset_train,
            eval_dataset=processed_dataset_eval,
            compute_metrics=compute_metrics_fn
        )
        
        try:
            # start training loop
            trainer.train()
        except:
            del trainer, tokenizer, training_args, dataset, processed_dataset_train, processed_dataset_eval
        
            gc.collect()
            torch.cuda.empty_cache()
        
        del trainer, tokenizer, training_args, dataset, processed_dataset_train, processed_dataset_eval
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    sweep_config = {
    'method': 'grid', 
    'metric': {  # This is the metric we are interested in maximizing
      'name': 'f1_score',
      'goal': 'maximize'   
    },
    # Paramaters and parameter values we are sweeping across
    'parameters': {
        'model_name': {
            #'values': ['michiyasunaga/BioLinkBERT-large']
            #'values': ['sultan/BioM-BERT-PubMed-PMC-Large']
            #'values':['microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'],
            'values':['microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract'],
            #'values': [
            #    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
            #    'michiyasunaga/BioLinkBERT-large',
            #    'sultan/BioM-BERT-PubMed-PMC-Large',
            #    'microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract',
            #    ]
        },
        'learning_rate': {
            'values': [1e-5, 2e-5, 3e-5, 5e-5, 7e-5]
        },
        'batch_size': {
            'values':[4]
        },
        'epochs':{
            'values': [3, 5, 10]
        },
        'warmup_ratio':{
            'values': [0.1, 0.5]
        },
        'weight_decay':{
            'values' : [0]
         }
    }
}

    sweep_id = wandb.sweep(sweep_config, project='DUVEL_finetune')
    wandb.agent(sweep_id, train)
