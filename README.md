# ILLUMINER: <u>I</u>nstruction-tuned <u>L</u>arge <u>L</u>ang<u>U</u>age <u>M</u>odels as Few-shot <u>IN</u>tent Classifier and Slot Fill<u>ER</u>

State-of-the-art intent classification (IC) and slot filling (SF) methods often rely on data-intensive deep learning 
models, limiting their practicality for industry applications. Large language models on the other hand, particularly
instruction-tuned models (Instruct-LLMs), exhibit remarkable zero-shot performance across various natural language
tasks. This study evaluates Instruct-LLMs on popular benchmark datasets for IC and SF, emphasizing their capacity
to learn from fewer examples. We introduce ILLUMINER, an approach framing IC and SF as language generation
tasks for Instruct-LLMs, with a more efficient SF-prompting method compared to prior work. 

A comprehensive comparison with multiple baselines shows that our approach, using the FLAN-T5 11B model, outperforms the
state-of-the-art joint IC+SF method and in-context learning with GPT3.5 (175B), particularly in slot filling by 11.1–32.2
percentage points. Additionally, our in-depth ablation study demonstrates that parameter-efficient fine-tuning requires
less than 6% of training data to yield comparable performance with traditional full-weight fine-tuning.

## How to run the evaluation service?

### Step 1: Prerequisites

Before running the evaluation service, please make sure your environment has all the requirements installed. For this, run: 
```shell
git clone https://github.com/paramita-mirza/illuminer.git
cd illuminer
pip install -r requirements.txt

# we need to install 🤗 Transformers and PEFT from source
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/peft
```

### Step 2: Prepare data
```shell
python -m evaluation.builder.data_builder_dsm
python -m evaluation.builder.data_builder_multiwoz
```

### Step 3: MLflow tracking

Set the MLflow tracking server URI:
- `cd evaluation/configs`
- In `main.yaml`, set the `mlflow_tracking_uri` with `http://<host>:<port>`

By default, it will use a local tracking server at `http://localhost:5000/`
- ``cd illuminer``
- ``mlflow server --host 127.0.0.1 --port 5000`` to run a local tracking server (you can see `mlruns` and `mlartifacts` with metadata of all the experiments at the root directory)
- Open `http://localhost:5000/` on your browser to view the runs

### Step 4: Run the experiment

```shell
python -m evaluation.run +data=<eval_data> +prompt=<eval_prompt> model=<llm>
```
---
**NOTE**
- `data` and `prompt` are mandatory configs without any default values, `model` is `google/flan-t5-xxl.yaml` by default
- Config options for:
  * `data` are in `evaluation/configs/data`
  * `prompt` are in `evaluation/configs/prompt`
  * `model` are in `evaluation/configs/model`
---
**IMPORTANT**
- `data` and `prompt` types must match:
  * `<dataset>-domain` with `zero/few-shot-dc-*`
  * `<dataset>-intent` with `zero/few-shot-ic-*`
  * `<dataset>-slot` with `zero/few-shot-sf-*` (`sf-sp` for slot filling with single-prompting and `sf-mp` for slot filling with multiple-prompting methods)
  * `<dataset>-dst` with `zero/few-shot-dst-*`

Sample evaluation runs for COLING 2024 are in `run_coling_2024.sh`.

### Parameter-efficient fine-tuning

```shell
python -m fine_tuning.run +data=<eval_data> +prompt=<eval_prompt> model=<llm> peft=<peft>
```
---
**NOTE**
- `data` and `prompt` are mandatory configs without any default values, `model` is `google/flan-t5-xxl.yaml` by default, `peft` is `lora.yaml` by default
- Config options for:
  * `data` are in `evaluation/configs/data`
  * `prompt` are in `evaluation/configs/prompt`
  * `model` are in `evaluation/configs/model`
  * `peft` are in `evaluation/configs/peft`
---
**IMPORTANT**
- `data` and `prompt` types must match:
  * `<dataset>-domain` with `zero/few-shot-dc-*`
  * `<dataset>-intent` with `zero/few-shot-ic-*`
  * `<dataset>-slot` with `zero/few-shot-sf-*` (`sf-sp` for slot filling with single-prompting and `sf-mp` for slot filling with multiple-prompting methods)
- When `peft=prompt_tuning`, the configuration for `prompt` must be either `dc-pt`, `ic-pt`, `sf-sp-pt` or `sf-mp-pt`
- Default values for `trainer`, such as `learning_rate` and `num_train_epochs` can be overridden, e.g., `trainer.num_train_epochs=10`
- Default values for `model`, such as `device` and `cache_dir` can be overridden, e.g., `model.device=mps`