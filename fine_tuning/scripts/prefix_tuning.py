from transformers import (
    AutoTokenizer,
    default_data_collator,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
)
from peft import get_peft_model, PrefixTuningConfig, TaskType
import torch
from datasets import Dataset
import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm



def run(data_path, model_out_path):
    device = "cuda"
    model_name_or_path = "google/flan-t5-xxl"
    tokenizer_name_or_path = "google/flan-t5-xxl"

    # data_path = "data/coling/ic/ic_snips_no_labels_no_instruction.json"
    # model_out_path = "/raid/s3/pm_tmp/adapters/prefix_tuning/ic_snips_flant5"

    text_column = "input"
    label_column = "output"
    max_input_length = 128
    # max_target_length = 10  # for IC
    max_target_length = 100  # for SF
    lr = 1e-2
    # num_epochs = 5  # for IC
    num_epochs = 20  # for SF
    batch_size = 8


    dataset = Dataset.from_json(data_path)

    print(dataset)

    # data preprocessing
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )


    train_dataset = processed_datasets

    print(train_dataset)

    # creating model
    peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir="/raid/s3/pm_tmp", device_map="auto")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # print(model)

    train_dataloader = DataLoader(
        dataset=train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )


    # training and evaluation
    # model = model.to(device)
    data_name = data_path.split("/")[-1].split("_")[1]
    training_args = Seq2SeqTrainingArguments(
        "out",
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_strategy="epoch",
        save_strategy="no",
        report_to=["wandb"],
        run_name=f"sf_prefix_tuning_{data_name}",
        predict_with_generate=True,
        generation_config=GenerationConfig(max_length=max_target_length),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    model.save_pretrained(model_out_path)
    tokenizer.save_pretrained(model_out_path)


if __name__ == "__main__":
    # data_paths = [
    #     "data/coling/ic/ic_snips_no_labels_no_instruction.json",
    #     "data/coling/ic/ic_multi_woz_no_labels_no_instruction.json",
    #     "data/coling/ic/ic_amz_en_no_labels_no_instruction.json"
    # ]
    # model_out_paths = [
    #     "/raid/s3/pm_tmp/adapters/prefix_tuning/ic_snips_flant5",
    #     "/raid/s3/pm_tmp/adapters/prefix_tuning/ic_multi_woz_flant5",
    #     "/raid/s3/pm_tmp/adapters/prefix_tuning/ic_amz_en_flant5"
    # ]

    data_paths = [
        "data/coling/sf/sf_snips_no_labels_no_instruction.json",
        "data/coling/sf/sf_multi_woz_no_labels_no_instruction.json",
        "data/coling/sf/sf_amz_en_no_labels_no_instruction.json"
    ]
    model_out_paths = [
        "/raid/s3/pm_tmp/adapters/prefix_tuning/sf_snips_flant5",
        "/raid/s3/pm_tmp/adapters/prefix_tuning/sf_multi_woz_flant5",
        "/raid/s3/pm_tmp/adapters/prefix_tuning/sf_amz_en_flant5"
    ]


    for data_path, model_out_path in zip(data_paths, model_out_paths):
        run(data_path, model_out_path)
