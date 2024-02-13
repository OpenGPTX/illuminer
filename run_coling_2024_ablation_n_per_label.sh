#!/bin/bash

datasets="multi_woz snips amz_en"

echo "Starting parameter-efficient-fine-tuning -- LoRA..."

models="flan-t5-xxl"

# Fine-tune intent classification, LoRA
for model in $models;
do
  python -m fine_tuning.run +data=10/multi_woz-intent-20 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/snips-intent-20 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/amz_en-intent-20 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500

  python -m fine_tuning.run +data=10/multi_woz-intent-50 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/snips-intent-50 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_en-intent-50 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500

  python -m fine_tuning.run +data=10/multi_woz-intent-100 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/snips-intent-100 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_en-intent-100 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500

  python -m fine_tuning.run +data=10/multi_woz-intent-full +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/snips-intent-full +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_en-intent-full +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
done

# Fine-tune slot filling, LoRA
for model in $models;
do
  python -m fine_tuning.run +data=10/multi_woz-slot-20 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/snips-slot-20 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/amz_en-slot-20 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500

  python -m fine_tuning.run +data=10/multi_woz-slot-50 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/snips-slot-50 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_en-slot-50 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500

  python -m fine_tuning.run +data=10/multi_woz-slot-100 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/snips-slot-100 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_en-slot-100 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500

  python -m fine_tuning.run +data=10/multi_woz-slot-full +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/snips-slot-full +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_en-slot-full +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
done

echo "Starting evaluation -- zero-shot and few-shot with fine-tuned models..."

models="flan-t5-xxl"

peft_methods="lora"

n_per_labels="20 50 100 full"

# Inference intent classification, LoRA
for dataset in $datasets;
do
  for model in $models;
  do
    for peft in $peft_methods;
    do
      for n in $n_per_labels;
      do
        python -m evaluation.run +data=${dataset}-intent +prompt=zero-shot-${prompt} model=sft/${dataset}/ic/${model}_${peft}_${n}
#        python -m evaluation.run +data=${dataset}-intent +prompt=few-shot-${prompt} model=sft/${dataset}/ic/${model}_${peft}_${n}
      done
    done
  done
done

# Inference slot filling, LoRA
for dataset in $datasets;
do
  for model in $models;
  do
    for peft in $peft_methods;
    do
      for n in $n_per_labels;
      do
        python -m evaluation.run +data=${dataset}-slot +prompt=zero-shot-sf-sp-001 model=sft/${dataset}/sf-sp/${model}_${peft}_${n}
#        python -m evaluation.run +data=${dataset}-slot +prompt=few-shot-sf-sp-001 model=sft/${dataset}/sf-sp/${model}_${peft}_${n}
      done
    done
  done
done
