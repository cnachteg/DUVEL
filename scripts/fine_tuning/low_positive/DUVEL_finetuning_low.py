#import ray
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
    load_from_disk,
    Features, 
    ClassLabel, 
    Value,
)

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
os.environ['WANDB_LOG_MODEL']="true"
os.environ['WANDB_PROJECT']='DUVEL_finetune'
os.environ['TOKENIZERS_PARALLELISM']="true"

num_gpus = torch.cuda.device_count()

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def train(config=None):
    with wandb.init(config=config, project='DUVEL_finetune', name=f'BiomedBERT_low_{config["round"]}'):
        
        # set configuration
        config = wandb.config
        tokenizer = AutoTokenizer.from_pretrained(config.model_name) # use_fast=True)
        dataset = load_dataset("cnachteg/duvel",use_auth_token=True)
        train_dataset = load_from_disk(f"../../data/low_positive/{config.round}")
        test_dataset = dataset['test']
        eval_dataset = dataset['validation']

        def compute_metrics_fn(eval_preds):
            table = wandb.Table(
                    columns=['id', 'sentence', 'prediction', 'truth']
                    )
            logits, labels, inputs = eval_preds
            predictions = np.argmax(logits, axis=-1)
            recall = evaluate.load('recall')
            precision = evaluate.load('precision')
            f1 = evaluate.load('f1')
            wandb.log({
                'recall': recall.compute(predictions=predictions, references=labels)['recall'],
                'precision':precision.compute(predictions=predictions, references=labels)['precision'],
                'f1_score':f1.compute(predictions=predictions, references=labels)['f1']
                })
            _id = 0
            for pred, label, inp in zip(predictions, labels, inputs):
                sentence = tokenizer.decode(inp).replace('[CLS]','').replace('[SEP]','').replace('[PAD]','')
                table.add_data(_id, sentence, pred, label)
                _id += 1

            wandb.log({"predictions":table})

            return {
                'recall': recall.compute(predictions=predictions, references=labels)['recall'],
                'precision':precision.compute(predictions=predictions, references=labels)['precision'],
                'f1_score':f1.compute(predictions=predictions, references=labels)['f1']
                }
        
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        #model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        
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
        processed_dataset_test = test_dataset.map(
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
            output_dir='DUVEL_finetune',
            report_to='wandb',  # Turn on Weights & Biases logging
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate*num_gpus,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=8,
            save_strategy='epoch',
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            remove_unused_columns=True,
            load_best_model_at_end=True,
            metric_for_best_model='f1_score',
            greater_is_better=True,
            include_inputs_for_metrics=True,
            seed=42,
            do_predict=True,
            do_train=True,
            do_eval=False
        )


        # define training loop
        trainer = Trainer(
            model_init=model_init,
            #model=model,
            args=training_args,
            data_collator=DefaultDataCollator(),
            train_dataset=processed_dataset_train,
            eval_dataset=processed_dataset_test,
            compute_metrics=compute_metrics_fn
        )
        
        try:
            # start training loop
            trainer.train()
            #trainer.evaluate()
        except:
            del trainer, tokenizer, training_args, dataset, processed_dataset_train, processed_dataset_test
        
            gc.collect()
            torch.cuda.empty_cache()
        
        del trainer, tokenizer, training_args, dataset, processed_dataset_train, processed_dataset_test
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    for i in range(5):
        config = {
        'batch_size': 4,
        #'model_name': 'michiyasunaga/BioLinkBERT-large',
        #'model_name': 'sultan/BioM-BERT-PubMed-PMC-Large',
        'model_name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        'learning_rate': 1e-5,
        #'epochs': 3,
        'epochs' : 5,
        'warmup_ratio' : 0.5,
        'weight_decay': 0,
        'round' : i

        }
        train(config)
