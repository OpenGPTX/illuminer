#!/bin/bash

datasets="multi_woz snips amz_en"

echo "Starting evaluation -- zero-shot and few-shot..."

models="google/flan-t5-xxl
        bigscience/bloomz-7b1
        lmsys/vicuna-13b-v1.5
        WizardLM/WizardLM-13B-V1.2
        tiiuae/falcon-7b-instruct"

prompts_domain="dc-001"
prompts_intent="ic-001 ic-002 ic-003 ic-004"
prompts_slot="sf-sp-001 sf-mp-001"

# Inference domain classification, non-fine-tuned
#for dataset in $datasets;
#do
#  for model in $models;
#  do
#    for prompt in $prompts_domain;
#    do
#      python -m evaluation.run +data=${dataset}-domain +prompt=zero-shot-${prompt} model=${model}
#      python -m evaluation.run +data=${dataset}-domain +prompt=few-shot-${prompt} model=${model}
#    done
#  done
#done

# Inference intent classification, non-fine-tuned
for dataset in $datasets;
do
  for model in $models;
  do
    for prompt in $prompts_intent;
    do
      python -m evaluation.run +data=${dataset}-intent +prompt=zero-shot-${prompt} model=${model}
      python -m evaluation.run +data=${dataset}-intent +prompt=few-shot-${prompt} model=${model}
    done
  done
done

# Inference slot filling, non-fine-tuned
for dataset in $datasets;
do
  for model in $models;
  do
    for prompt in $prompts_slot;
    do
      python -m evaluation.run +data=${dataset}-slot +prompt=zero-shot-${prompt} model=${model}
      python -m evaluation.run +data=${dataset}-slot +prompt=few-shot-${prompt} model=${model}
    done
  done
done

echo "Starting parameter-efficient-fine-tuning -- LoRA..."

models="flan-t5-xxl t5-large
        bloomz-7b1 bloom-7b1
        vicuna-13b-v1.5
        WizardLM-13B-V1.2
        falcon-7b-instruct falcon-7b"

# Fine-tune domain classification, LoRA
#for model in $models;
#do
#  python -m fine_tuning.run +data=10/multi_woz-domain-10 +prompt=dc model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
#  python -m fine_tuning.run +data=10/snips-domain-10 +prompt=dc model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
#  python -m fine_tuning.run +data=10/amz_en-domain-10 +prompt=dc model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
#done

# Fine-tune intent classification, LoRA
for model in $models;
do
  python -m fine_tuning.run +data=10/multi_woz-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/snips-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/amz_en-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
done

# Fine-tune slot filling, LoRA
for model in $models;
do
  python -m fine_tuning.run +data=10/multi_woz-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/snips-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/amz_en-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500

  python -m fine_tuning.run +data=10/multi_woz-slot-10 +prompt=sf-mp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/snips-slot-10 +prompt=sf-mp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/amz_en-slot-10 +prompt=sf-mp model=${model} main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
done

echo "Starting evaluation -- zero-shot and few-shot with fine-tuned models..."

models="flan-t5-xxl t5-large
        bloomz-7b1 bloom-7b1
        vicuna-13b-v1.5
        WizardLM-13B-V1.2
        falcon-7b-instruct falcon-7b"

prompts_domain="dc-001"
prompts_intent="ic-001 ic-002 ic-003 ic-004"

peft_methods="lora"

# Inference domain classification, LoRA
for dataset in $datasets;
do
  for model in $models;
  do
    for prompt in $prompts_domain;
    do
      for peft in $peft_methods;
      do
        python -m evaluation.run +data=${dataset}-domain +prompt=zero-shot-${prompt} model=sft/${dataset}/dc/${model}_${peft}
#        python -m evaluation.run +data=${dataset}-domain +prompt=few-shot-${prompt} model=sft/${dataset}/dc/${model}_${peft}
      done
    done
  done
done

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
#        python -m evaluation.run +data=${dataset}-intent +prompt=few-shot-${prompt} model=sft/${dataset}/ic/${model}_${peft}
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
      python -m evaluation.run +data=${dataset}-slot +prompt=zero-shot-sf-sp-001 model=sft/${dataset}/sf-sp/${model}_${peft}
#      python -m evaluation.run +data=${dataset}-slot +prompt=few-shot-sf-sp-001 model=sft/${dataset}/sf-sp/${model}_${peft}

      python -m evaluation.run +data=${dataset}-slot +prompt=zero-shot-sf-mp-001 model=sft/${dataset}/sf-mp/${model}_${peft}
#      python -m evaluation.run +data=${dataset}-slot +prompt=few-shot-sf-mp-001 model=sft/${dataset}/sf-mp/${model}_${peft}
    done
  done
done
