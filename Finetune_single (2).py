import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import (
    MllamaForConditionalGeneration,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    Trainer
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import yaml
from sklearn.model_selection import train_test_split
from collections import Counter

def set_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_and_preprocess_data(task, config):
    
    if task == "glue":
        df = pd.read_csv(config["path_glue"])
        df = df.sample(frac=1, random_state=config['random_seed']).reset_index(drop=True)
        df.columns = ['text', 'label'] 
        #df = df.head(5000)
        
        df_test = pd.read_csv(config["path_glue_eval"])
        df_test = df_test.sample(frac=1, random_state=config['random_seed']).reset_index(drop=True)
        df_test.columns = ['text', 'label'] 
        #df_test = df_test.head(500)
        
    elif task == "quora":
        df = pd.read_csv(config["path_quora"])
        df = df.sample(frac=1, random_state=config['random_seed']).reset_index(drop=True)
        df.columns = ['text', 'label'] 
        #df = df.head(5000)

        df_test = pd.read_csv(config["path_quora_eval"])
        df_test = df_test.sample(frac=1, random_state=config['random_seed']).reset_index(drop=True)
        df_test.columns = ['text', 'label'] 
        #df_test = df_test.head(500)
    
    elif task == "sentiment":

        df = pd.read_csv(config["path_sentiment"])
        df = df.sample(frac=1, random_state=config['random_seed']).reset_index(drop=True)
        df.columns = ['text', 'label'] 
        #df = df.head(5000)

        df_test = pd.read_csv(config["path_sentiment_eval"])
        df_test = df_test.sample(frac=1, random_state=config['random_seed']).reset_index(drop=True)
        df_test.columns = ['text', 'label'] 
        #df_test = df_test.head(500)
        
    
    else:
        # Fallback case: handle if 'task' is not one of the expected values
        raise ValueError(f"Unsupported task type:")
    #print(len(df), len(df_test))
    return df, df_test


def generate_prompt(data_point, task, config, is_test=False):
    # Define prompt based on the specified system prompt (SP) and task
    if config['SP'] == "No_SP":
        prompt = f"""
text: {data_point["text"]}
label: {"" if is_test else data_point["label"]}""".strip()
    
    elif config['SP'] == "Same_SP":
        prompt = f"""
"Analyze the task text and assign the correct class label.\n"
text: {data_point["text"]}
label: {"" if is_test else data_point["label"]}""".strip()
    
    elif config['SP'] == "Different_SP":
        if task == "quora":
            prompt = f"""
"Evaluate each sentence pair: return 'True' if they are paraphrases, 'False' if not.\n"
text: {data_point["text"]}
label: {"" if is_test else data_point["label"]}""".strip()
    
        elif task == "sentiment":
            prompt = f"""
"Classify the text into Normal, Depression, Anxiety, Bipolar, and return the answer as the corresponding mental health disorder label.\n"
text: {data_point["text"]}
label: {"" if is_test else data_point["label"]}""".strip()
    
        elif task == "glue":
            prompt = f"""
"Classify if each sentence is grammatically correct or not and return the answer as either 'acceptable'  or 'unacceptable' based on English language standards.\n"
text: {data_point["text"]}
label: {"" if is_test else data_point["label"]}""".strip()

    return prompt


def prepare_datasets(X_train, X_eval, task, tokenizer, config):
   

    X_train['text'] = X_train.apply(lambda row: generate_prompt(row, task=task, config=config), axis=1)
    X_eval['text'] = X_eval.apply(lambda row: generate_prompt(row, task=task, config=config), axis=1)
    
    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)
    
    def tokenize_add_label(sample, max_length=200):
        prompt = sample["text"]
        label = str(sample["label"])
    
        # Encode prompt and label separately without padding
        prompt_ids = tokenizer.encode(tokenizer.bos_token + prompt, add_special_tokens=False)
        label_ids = tokenizer.encode(label + tokenizer.eos_token, add_special_tokens=False)
    
        # Combine the encoded prompt and label
        combined_input = prompt_ids + label_ids
        
        # Pad to max_length
        combined_input = combined_input[:max_length]  # Truncate if longer than max_length
        combined_input += [tokenizer.pad_token_id] * (max_length - len(combined_input))  # Pad to max_length
    
         # Compute the attention mask: 1s until and including the first eos, then 0s
        eos_token_id = tokenizer.eos_token_id
        first_eos_index = next((i for i, token in enumerate(combined_input) if token == eos_token_id), len(combined_input))
        attention_mask = [1 if i <= first_eos_index else 0 for i in range(max_length)]
    
    
        # Generate labels: -100 for prompt tokens, keep label tokens
        labels = [-100] * len(prompt_ids) + label_ids
        labels += [-100] * (max_length - len(labels))
        labels = labels[:max_length]  
    
        # Pack into a dictionary format
        tokenized_output = {
            "input_ids": combined_input,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return tokenized_output

    
    tokenized_train_data = train_data.map(tokenize_add_label, remove_columns=list(train_data.features))
    tokenized_eval_data = eval_data.map(tokenize_add_label, remove_columns=list(eval_data.features))
    
    return tokenized_train_data, tokenized_eval_data

    
def load_model_and_tokenizer(config):
   
    model_name=config['base_model_name']

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float32
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(config['base_model_name'])
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def setup_training(config, model, train_data, eval_data):

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        optim=config['optim'],
        logging_steps=config['logging_steps'],
        learning_rate=1e-4,
        weight_decay=config['weight_decay'],
        report_to=config['report_to'],
        evaluation_strategy=config['evaluation_strategy'],
        eval_steps=config['eval_steps'], 
        logging_dir=config['logging_dir'], 
        save_steps=config['save_steps'],  
        save_total_limit=config['save_total_limit'],
        
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    return trainer

def main(config_path):

    #config_path="./config.yaml"
    config = load_config(config_path)
    set_environment()
    
    task=config["task"]
        
    X_train, X_eval = load_and_preprocess_data(task, config)
    model, tokenizer = load_model_and_tokenizer(config)
    
    train_data, eval_data = prepare_datasets(X_train, X_eval, task, tokenizer, config)

    if config['FT_type']=="LoRA":
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=find_all_linear_names(model),
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        
    trainer = setup_training(config, model, train_data, eval_data)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate LLM model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
