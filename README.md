# AMA: Attractive Metadata Attack

---

## Overview

This repository provides the official implementation of Attractive Metadata Attack (AMA). AMA crafts “attractive” tool metadata to induce LLM agents to invoke malicious tools during tool selection, enabling systematic evaluation and study of agent security.

Paper: Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools (NeurIPS 2025, Poster)  
Authors: Kanghua Mo, Li Hu, Yucheng Long, Zhihao Li  
Poster link: https://neurips.cc/virtual/2025/loc/san-diego/poster/116046

---

## Installation

- Python 3.11 is required.
- We recommend `uv` for environment management.

Using uv:

```bash
uv venv --python=3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Configuration (.env)

All API access is configured via environment variables. Edit the `.env` file at the repository root (a placeholder is provided):

```env
# OpenAI-compatible (GPT)
GPT_API_BASE=
GPT_API_KEY=

# Xinference (optional)
XINFERENCE_API_BASE=
XINFERENCE_API_KEY=
```

Guidance:
- Set `GPT_API_BASE` to your OpenAI-compatible endpoint (e.g., `https://api.openai.com/v1`) and `GPT_API_KEY` to the corresponding key.
- If you plan to use Xinference, set `XINFERENCE_API_BASE` to your Xinference REST endpoint (e.g., `http://<host>:<port>/v1`) and `XINFERENCE_API_KEY` if required by your deployment.
- The code automatically loads these values at runtime; use real credentials for actual runs. To evaluate locally served models (e.g., Qwen3, Qwen2.5) or non-GPT commercial endpoints, set up Xinference and configure `XINFERENCE_API_BASE`/`XINFERENCE_API_KEY`. For launch instructions, see https://github.com/xorbitsai/inference.

---

## Optional: Deploy Xinference and Configure API

You can run AMA against models served by Xinference. After deployment, update `.env` with your Xinference endpoint and key.

Then set in `.env` (example):

```env
XINFERENCE_API_BASE=http://<your-ip>:9997/v1
XINFERENCE_API_KEY=sk-xxxx
```

### Local model configuration examples

- See `config/local_llm.yml` for example settings to run locally served models via Xinference or other OpenAI-compatible endpoints.
- See `main_attacker.py` for how CLI arguments map to runtime parameters; align your config values and command-line flags accordingly when running local models.

---

## Quickstart

Below shows a full evaluation pipeline using `gpt-4o-mini` as an example.

1) Build the privacy dataset

```bash
python3 -m construct_privacy_info.construct_memory
```

2) Generate privacy memory parameters

```bash
python3 -m construct_privacy_info.generate_privacy_parameter --llm_name gpt-4o-mini
```

3) Generate adversarial (attack) tool metadata

```bash
python3 generate_tool_metadata.py --llm_name gpt-4o-mini --attack_type target -t 0.95 --lambda_weight 0.5
```

4) Run the main attack pipeline (via config)

```bash
python3 scripts/run.py --cfg_path config/gpt.yml
```

For other models (local or non-GPT commercial endpoints), first deploy Xinference and set the following in your `.env`:

```env
XINFERENCE_API_BASE=
XINFERENCE_API_KEY=
```

Then run the local configuration to evaluate AMA, AMA combined with other attacks, and AMA under defense settings:

```bash
python3 scripts/run.py --cfg_path config/local_llm.yml
```

For details on local model settings and scenarios, see `config/local_llm.yml`.

Equivalent low-level command (for reference):

```bash
python -u main_attacker.py \
   --llm_name gpt-4o-mini \
   --attack_type pure \
   --use_backend None \
   --attacker_tools_path data/generated_target_attack_tools.jsonl \
   --res_file logs/target/gpt-4o-mini/attack/pure/steal_parmas_num_5/test.xlsx \
   --run_tools_num 2 \
   --steal_parmas_num 5 \
   --task_num 5 \
   --parallel_num 20 \
   --attack_tool_type target \
   --direct_prompt_injection
```

Outputs are saved as Excel files under paths like:

```
logs/target/gpt-4o-mini/attack/pure/steal_parmas_num_5/
```

Untargeted variant (optional):

```bash
python3 generate_tool_metadata.py --llm_name qwen2.5-instruct-32b --attack_type untarget -t 0.8 --lambda_weight 0.5
python3 scripts/run.py --cfg_path config/qwen3_untarget.yml
```

### Local run (via `config/local_llm.yml`)

To evaluate locally served models (via Xinference):

1) Start Xinference and set `.env` (`XINFERENCE_API_BASE`, `XINFERENCE_API_KEY`).
2) Run with the provided local configuration:

```bash
python3 scripts/run.py --cfg_path config/local_llm.yml
```

The YAML includes example local models (e.g., `qwen3`, `qwen2.5-instruct-32b`, `llama-3.3-instruct`, `gemma-3-it`) and attack settings.

---

## Acknowledgments

We thank the Agent Security Bench (ASB) project for inspiration and resources:
- https://github.com/agiresearch/ASB

BibTeX (ASB):

```bibtex
@inproceedings{
   zhang2025agent,
   title={Agent Security Bench ({ASB}): Formalizing and Benchmarking Attacks and Defenses in {LLM}-based Agents},
   author={Hanrong Zhang and Jingyuan Huang and Kai Mei and Yifei Yao and Zhenting Wang and Chenlu Zhan and Hongwei Wang and Yongfeng Zhang},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=V4y0CpX4hK}
}
```

---

## Citation

If you find this repository helpful, please cite our NeurIPS 2025 poster:

```
@inproceedings{mo2025ama,
   title={Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools},
   author={Mo, Kanghua and Hu, Li and Long, Yucheng and Li, Zhihao},
   booktitle={Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2025), Poster},
   year={2025},
   url={https://neurips.cc/virtual/2025/loc/san-diego/poster/116046}
}
```
---