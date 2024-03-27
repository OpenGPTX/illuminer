#!/bin/bash

echo "Starting evaluation -- joint IC+SF..."

#models="flan-t5-xxl"
#models="mt5-xxl"
models="mt0-xxl"
languages="de fr it es"

# Fine-tune intent classification, LoRA
for model in $models;
do
  for lang in $languages;
  do
#    python -m fine_tuning.run +data=10/amz_${lang}-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
    python -m fine_tuning.run +data=10/amz_${lang}-intent-10_${lang} +prompt=ic_${lang} model=${model}-mt main.model_output_dir=/raid/s3/pm_tmp/lora_adapters/ peft=lora trainer.max_steps=500 trainer.learning_rate=5e-4
  done
done

## Fine-tune slot filling, LoRA
#for model in $models;
#do
#  for lang in $languages;
#  do
#    python -m fine_tuning.run +data=10/amz_${lang}-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
#    python -m fine_tuning.run +data=10/amz_${lang}-slot-10_${lang} +prompt=sf-sp_${lang} model=${model}-mt main.model_output_dir=/raid/s3/pm_tmp/lora_adapters/ peft=lora trainer.max_steps=500
#  done
#done

# models="flan-t5-xxl mt5-xxl polylm-multialpaca-13b"
#models="mt5-xxl"
models="mt0-xxl"
languages="de fr it es"

## Inference intent classification, LoRA
#for model in $models;
#do
#  for lang in $languages;
#  do
##    python -m evaluation.run +data=amz_${lang}-intent +prompt=zero-shot-ic-001 model=sft/amz_${lang}/ic/${model}_lora main.mlflow_tracking_uri=http://localhost:5077/
#    python -m evaluation.run +data=amz_${lang}-intent_${lang} +prompt=zero-shot-ic-001_${lang} model=sft/amz_${lang}/ic/${model}-mt_lora_${lang} main.mlflow_tracking_uri=http://localhost:5077/
#  done
#done

## Inference slot filling, LoRA
#for model in $models;
#do
#  for lang in $languages;
#  do
#    python -m evaluation.run +data=amz_${lang}-slot +prompt=zero-shot-sf-sp-001 model=sft/amz_${lang}/sf-sp/${model}_lora main.mlflow_tracking_uri=http://localhost:5077/
#    python -m evaluation.run +data=amz_${lang}-slot_${lang} +prompt=zero-shot-sf-sp-001_${lang} model=sft/amz_${lang}/sf-sp/${model}-mt_lora_${lang} main.mlflow_tracking_uri=http://localhost:5077/
#  done
#done
