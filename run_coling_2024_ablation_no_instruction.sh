#!/bin/bash

datasets="multi_woz snips amz_en"

echo "Starting parameter-efficient-fine-tuning -- LoRA, without instructions..."

models="flan-t5-xxl"

# Fine-tune intent classification, LoRA
for model in $models;
do
  python -m fine_tuning.run +data=10/multi_woz-intent-10 +prompt=ic-no-instruction model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/snips-intent-10 +prompt=ic-no-instruction model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/amz_en-intent-10 +prompt=ic-no-instruction model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
done

# Fine-tune slot filling, LoRA
for model in $models;
do
  python -m fine_tuning.run +data=10/multi_woz-slot-10 +prompt=sf-sp-no-instruction model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/snips-slot-10 +prompt=sf-sp-no-instruction model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/amz_en-slot-10 +prompt=sf-sp-no-instruction model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
done

echo "Starting evaluation -- zero-shot and few-shot with fine-tuned models..."

models="flan-t5-xxl"

prompts_intent="ic-001 ic-001-no-instruction"
prompts_slot="sf-sp-001 sf-sp-001-no-instruction"
peft_methods="lora"


# Inference intent classification, LoRA
for dataset in $datasets;
do
  for model in $models;
  do
    for prompt in $prompts_intent;
    do
      for peft in $peft_methods;
      do
        python -m evaluation.run +data=${dataset}-intent +prompt=zero-shot-${prompt} model=sft/${dataset}/ic/${model}_${peft}
        python -m evaluation.run +data=${dataset}-intent +prompt=zero-shot-${prompt} model=sft/${dataset}/ic/${model}_${peft}_no_instruction
      done
    done
  done
done

# Inference slot filling, LoRA
for dataset in $datasets;
do
  for model in $models;
  do
    for prompt in $prompts_slot;
    do
      for peft in $peft_methods;
      do
        python -m evaluation.run +data=${dataset}-slot +prompt=zero-shot-${prompt} model=sft/${dataset}/sf-sp/${model}_${peft}
        python -m evaluation.run +data=${dataset}-slot +prompt=zero-shot-${prompt} model=sft/${dataset}/sf-sp/${model}_${peft}_no_instruction
      done
    done
  done
done
