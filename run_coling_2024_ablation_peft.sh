#!/bin/bash

datasets="multi_woz snips amz_en"

echo "Starting parameter-efficient-fine-tuning (IA3, prefix tuning, prompt tuning)..."

models="flan-t5-xxl"

# Fine-tune intent classification, IA3, prefix tuning, prompt tuning
for model in $models;
do
  python -m fine_tuning.run +data=10/snips-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=ia3 trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/snips-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=prefix_tuning trainer.max_steps=-1 trainer.num_train_epochs=10 trainer.learning_rate=1e-2
  python -m fine_tuning.run +data=10/snips-intent-10 +prompt=ic-pt model=${model} main.model_output_dir=lora_adapters/ peft=prompt_tuning trainer.max_steps=-1 trainer.num_train_epochs=10 trainer.learning_rate=1e-2

  python -m fine_tuning.run +data=10/multi_woz-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=ia3 trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/multi_woz-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=prefix_tuning trainer.max_steps=-1 trainer.num_train_epochs=10 trainer.learning_rate=1e-2
  python -m fine_tuning.run +data=10/multi_woz-intent-10 +prompt=ic-pt model=${model} main.model_output_dir=lora_adapters/ peft=prompt_tuning trainer.max_steps=-1 trainer.num_train_epochs=10 trainer.learning_rate=1e-2

  python -m fine_tuning.run +data=10/amz_en-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=ia3 trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_en-intent-10 +prompt=ic model=${model} main.model_output_dir=lora_adapters/ peft=prefix_tuning trainer.max_steps=500 trainer.learning_rate=1e-2
  python -m fine_tuning.run +data=10/amz_en-intent-10 +prompt=ic-pt model=${model} main.model_output_dir=lora_adapters/ peft=prompt_tuning trainer.max_steps=500 trainer.learning_rate=1e-2
done

# Fine-tune slot filling, IA3, prefix tuning, prompt tuning
for model in $models;
do
  python -m fine_tuning.run +data=10/snips-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=ia3 trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/snips-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=prefix_tuning trainer.max_steps=-1 trainer.num_train_epochs=10 trainer.learning_rate=1e-2
  python -m fine_tuning.run +data=10/snips-slot-10 +prompt=sf-sp-pt model=${model} main.model_output_dir=lora_adapters/ peft=prompt_tuning trainer.max_steps=-1 trainer.num_train_epochs=10 trainer.learning_rate=1e-2

  python -m fine_tuning.run +data=10/multi_woz-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=ia3 trainer.max_steps=-1 trainer.num_train_epochs=10
  python -m fine_tuning.run +data=10/multi_woz-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=prefix_tuning trainer.max_steps=-1 trainer.num_train_epochs=10 trainer.learning_rate=1e-2
  python -m fine_tuning.run +data=10/multi_woz-slot-10 +prompt=sf-sp-pt model=${model} main.model_output_dir=lora_adapters/ peft=prompt_tuning trainer.max_steps=-1 trainer.num_train_epochs=10 trainer.learning_rate=1e-2

  python -m fine_tuning.run +data=10/amz_en-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=ia3 trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_en-slot-10 +prompt=sf-sp model=${model} main.model_output_dir=lora_adapters/ peft=prefix_tuning trainer.max_steps=500 trainer.learning_rate=1e-2
  python -m fine_tuning.run +data=10/amz_en-slot-10 +prompt=sf-sp-pt model=${model} main.model_output_dir=lora_adapters/ peft=prompt_tuning trainer.max_steps=500 trainer.learning_rate=1e-2
done

echo "Starting evaluation -- zero-shot and few-shot with fine-tuned models (IA3, prefix tuning, prompt tuning)..."

models="flan-t5-xxl"

peft_methods="ia3 prefix_tuning prompt_tuning"

# Inference intent classification, LoRA
for dataset in $datasets;
do
  for model in $models;
  do
    for peft in $peft_methods;
    do
      python -m evaluation.run +data=${dataset}-intent +prompt=zero-shot-ic-001 model=sft/${dataset}/ic/${model}_${peft}
#      python -m evaluation.run +data=${dataset}-intent +prompt=few-shot-ic-001 model=sft/${dataset}/ic/${model}_${peft}
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
    done
  done
done

peft_methods="lora ia3 prefix_tuning prompt_tuning"

# Inference intent classification, LoRA, cross-dataset
for model in $models;
do
  for peft in $peft_methods;
  do
    python -m evaluation.run +data=multi_woz-intent +prompt=zero-shot-ic-001 model=sft/snips/ic/${model}_${peft}
    python -m evaluation.run +data=multi_woz-intent +prompt=zero-shot-ic-001 model=sft/amz_en/ic/${model}_${peft}

    python -m evaluation.run +data=snips-intent +prompt=zero-shot-ic-001 model=sft/multi_woz/ic/${model}_${peft}
    python -m evaluation.run +data=snips-intent +prompt=zero-shot-ic-001 model=sft/amz_en/ic/${model}_${peft}

    python -m evaluation.run +data=amz_en-intent +prompt=zero-shot-ic-001 model=sft/snips/ic/${model}_${peft}
    python -m evaluation.run +data=amz_en-intent +prompt=zero-shot-ic-001 model=sft/multi_woz/ic/${model}_${peft}
  done
done

# Inference slot filling, LoRA, cross-dataset
for model in $models;
do
  for peft in $peft_methods;
  do
    python -m evaluation.run +data=multi_woz-slot +prompt=zero-shot-sf-sp-001 model=sft/snips/sf-sp/${model}_${peft}
    python -m evaluation.run +data=multi_woz-slot +prompt=zero-shot-sf-sp-001 model=sft/amz_en/sf-sp/${model}_${peft}

    python -m evaluation.run +data=snips-slot +prompt=zero-shot-sf-sp-001 model=sft/multi_woz/sf-sp/${model}_${peft}
    python -m evaluation.run +data=snips-slot +prompt=zero-shot-sf-sp-001 model=sft/amz_en/sf-sp/${model}_${peft}

    python -m evaluation.run +data=amz_en-slot +prompt=zero-shot-sf-sp-001 model=sft/snips/sf-sp/${model}_${peft}
    python -m evaluation.run +data=amz_en-slot +prompt=zero-shot-sf-sp-001 model=sft/multi_woz/sf-sp/${model}_${peft}
  done
done
