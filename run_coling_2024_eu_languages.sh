#!/bin/bash

echo "Starting evaluation -- multilinguality experiments..."

languages="de fr it es"

# Fine-tune intent classification, LoRA
python -m fine_tuning.run +data=10/amz_en-intent-10 +prompt=ic model=mt5-xxl peft=lora trainer.max_steps=500
python -m fine_tuning.run +data=10/amz_en-intent-10 +prompt=ic model=mt0-xxl peft=lora trainer.max_steps=500
python -m fine_tuning.run +data=10/amz_en-intent-10 +prompt=ic model=mt0-xxl-mt peft=lora trainer.max_steps=500 trainer.learning_rate=5e-4

for lang in $languages;
do
  python -m fine_tuning.run +data=10/amz_${lang}-intent-10 +prompt=ic model=flan-t5-xxl peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_${lang}-intent-10 +prompt=ic model=mt5-xxl peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_${lang}-intent-10 +prompt=ic model=mt0-xxl peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_${lang}-intent-10_${lang} +prompt=ic_${lang} peft=lora trainer.max_steps=500 trainer.learning_rate=5e-4
done

# Fine-tune slot filling, LoRA
python -m fine_tuning.run +data=10/amz_en-slot-10 +prompt=sf-sp model=mt5-xxl peft=lora trainer.max_steps=500
python -m fine_tuning.run +data=10/amz_en-slot-10 +prompt=sf-sp model=mt0-xxl peft=lora trainer.max_steps=500
python -m fine_tuning.run +data=10/amz_en-slot-10 +prompt=sf-sp model=mt0-xxl-mt peft=lora trainer.max_steps=500 trainer.learning_rate=5e-4

for lang in $languages;
do
  python -m fine_tuning.run +data=10/amz_${lang}-slot-10 +prompt=sf-sp model=flan-t5-xxl main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_${lang}-slot-10 +prompt=sf-sp model=mt5-xxl main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_${lang}-slot-10 +prompt=sf-sp model=mt0-xxl main.model_output_dir=lora_adapters/ peft=lora trainer.max_steps=500
  python -m fine_tuning.run +data=10/amz_${lang}-slot-10_${lang} +prompt=sf-sp_${lang} model=mt0-xxl-mt main.model_output_dir=/raid/s3/pm_tmp/lora_adapters/ peft=lora trainer.max_steps=500 trainer.learning_rate=5e-4
done

languages="de fr it es"

# Inference intent classification, LoRA
python -m evaluation.run +data=amz_en-intent +prompt=zero-shot-ic-001 model=sft/amz_en/ic/mt5-xxl_lora
python -m evaluation.run +data=amz_en-intent +prompt=zero-shot-ic-001 model=sft/amz_en/ic/mt0-xxl_lora
python -m evaluation.run +data=amz_en-intent +prompt=zero-shot-ic-001 model=sft/amz_en/ic/mt0-xxl-mt_lora

for lang in $languages;
do
  python -m evaluation.run +data=amz_${lang}-intent +prompt=zero-shot-ic-001 model=sft/amz_${lang}/ic/flan-t5-xxl_lora
  python -m evaluation.run +data=amz_${lang}-intent +prompt=zero-shot-ic-001 model=sft/amz_${lang}/ic/mt5-xxl_lora
  python -m evaluation.run +data=amz_${lang}-intent +prompt=zero-shot-ic-001 model=sft/amz_${lang}/ic/mt0-xxl_lora
  python -m evaluation.run +data=amz_${lang}-intent_${lang} +prompt=zero-shot-ic-001_${lang} model=sft/amz_${lang}/ic/mt0-xxl-mt_lora_${lang}
done

# Inference slot filling, LoRA
python -m evaluation.run +data=amz_en-slot +prompt=zero-shot-sf-sp-001 model=sft/amz_en/sf-sp/mt5-xxl_lora
python -m evaluation.run +data=amz_en-slot +prompt=zero-shot-sf-sp-001 model=sft/amz_en/sf-sp/mt0-xxl_lora
python -m evaluation.run +data=amz_en-slot +prompt=zero-shot-sf-sp-001 model=sft/amz_en/sf-sp/mt0-xxl-mt_lora

for lang in $languages;
do
  python -m evaluation.run +data=amz_${lang}-slot +prompt=zero-shot-sf-sp-001 model=sft/amz_${lang}/sf-sp/flan-t5-xxl_lora
  python -m evaluation.run +data=amz_${lang}-slot +prompt=zero-shot-sf-sp-001 model=sft/amz_${lang}/sf-sp/mt5-xxl_lora
  python -m evaluation.run +data=amz_${lang}-slot +prompt=zero-shot-sf-sp-001 model=sft/amz_${lang}/sf-sp/mt0-xxl_lora
  python -m evaluation.run +data=amz_${lang}-slot_${lang} +prompt=zero-shot-sf-sp-001_${lang} model=sft/amz_${lang}/sf-sp/mt0-xxl-mt_lora_${lang}
done
