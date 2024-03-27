#!/bin/bash

datasets="multi_woz snips amz_en"

echo "Starting evaluation -- joint IC+SF..."

models="flan-t5-xxl
        bloomz-7b1
        vicuna-13b-v1.5
        WizardLM-13B-V1.2
        falcon-7b-instruct"

# Inference joint IC+SF, LoRA
for dataset in $datasets;
do
  for model in $models;
  do
#    python -m evaluation.run +data=${dataset}-dst +prompt=zero-shot-dst-001 model=sft/${dataset}/dst/${model}_lora_001
#    python -m evaluation.run +data=${dataset}-dst +prompt=few-shot-dst-001 model=sft/${dataset}/dst/${model}_lora_001

    python -m evaluation.run +data=${dataset}-dst +prompt=zero-shot-dst-002 model=sft/${dataset}/dst/${model}_lora_002
#    python -m evaluation.run +data=${dataset}-dst +prompt=few-shot-dst-002 model=sft/${dataset}/dst/${model}_lora_002
  done
done

# Inference joint IC+SF with GPT3.5
models="openai/gpt-3.5-turbo-instruct"

for dataset in $datasets;
do
  for model in $models;
  do
#    python -m evaluation.run +data=${dataset}-dst +prompt=zero-shot-dst-001 model=${model}
#    python -m evaluation.run +data=${dataset}-dst +prompt=few-shot-dst-001 model=${model}

    python -m evaluation.run +data=${dataset}-dst +prompt=zero-shot-dst-002 model=${model}
    python -m evaluation.run +data=${dataset}-dst +prompt=few-shot-dst-002 model=${model}
  done
done
