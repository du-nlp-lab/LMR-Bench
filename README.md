# LMR-BENCH: Evaluating LLM Agent’s Ability on Reproducing Language Modeling Research


## Overview
*LMR-Bench* is a benchmark for evaluating large language model (LLM) agents on their ability to reproduce code from *NLP research papers*.
Given a *research paper*, a *code repo with masked methods*, and a *implementation instruction*, an LLM agent is tasked with generating *patch* code to correctly fill in the missing methods. The benchmark covers 28 tasks from 23 top-tier NLP papers across nine research categories, providing a systematic way to assess the scientific reasoning and code synthesis abilities of LLMs.


<!-- ### Abstract: 
 Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present \ours, a comprehensive benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from NLP research papers. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked methods, and instructions for implementing these methods.
We conduct extensive experiments in standalone and agent-based settings on state-of-the-art LLMs, evaluating the accuracy of unit tests and performing both LLM and human evaluation of code correctness.
Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLMs’ ability to autonomously reproduce scientific research. We will release our benchmark and code after publication.
-->


## Environment Setup
LMR-Bench requires Python ≥ 3.12.

We recommend using a virtual environment to avoid dependency conflicts.

**1. Clone the repository**:
```
git clone https://github.com/du-nlp-lab/LMR-Bench.git
cd LMR-Bench
```

The structure of the directory of LMR-Bench should be like the following:
```text
LMR-Bench/
├── benchmark/                        # Contains all datasets. Each project has a subfolder.
│   └── datasets
│       └── project_folder_1/
│           ├── repository_folder_1/
│           │   ├── ...                   # Masked code repo (Rewrite & Mask)
│           │   └── unit_test/
│           │       └── unit_test_1.py    # Unit test file (Annotated)
│           ├── info.json                 # Metadata for this benchmark task. (Annotated)
│           ├── Dockerfile                # Docker ENV for reproducibility. We've built and pushed Docker Images for each project to DOCKER HUB for faster evaluation.
│           └── golden_files/
│               └── ...                   # Reference solutions or golden outputs
│
├── evaluation/                       # Evaluation
│   ├── ...                 
├── generation/                       # Generation/inference (noagent & necessary scripts for OpenHands)
│   ├── NoAgent/
│   └── OpenHands/
├── results/                      
│   ├── human_evaluation/
│   ├── llm_as_a_judge_evaluation/
│   ├── logs/
│   └── unit_test_evaluation/
├── scripts/                          # Shell scripts
│   ├── no_agent_generation.sh
│   ├── no_agent_generation_claude.sh
│   ├── llm_as_a_judge_evaluation.sh
│   ├── unit_test_evaluation.sh
├── utils/                            # Utility modules
│   └── ...
└── requirements.txt

```

- `benchmark/dataset`: Contains all benchmark tasks. Each project is a subdirectory with its repository, metadata, unit tests, and golden/reference files.
- `evaluation/`: Main Python evaluation logic.
- `generation/`: Code for generation/inference (with or without agents), organized by method category. Add your agent under this folder if you want to test your performance. 
- `results/`: Output/results directories.
- `scripts/`: Shell scripts for automated/batch execution for running generation and evaluation.
- `utils/`: Utility modules and functions.

**2. Install dependencies**:
```
conda create -n lmrbench python=3.12
conda activate lmrbench
pip install -r requirements.txt
```

**3. Download the benchmark:**
The benchmark data used in our paper can be downloaded from [this link](https://drive.google.com/drive/folders/13k1NtiSXd_mypXWmbvn0jjd0tFtHbiou?usp=sharing). 
Download the data under the folder benchmark/ and then unzip
```
cd benchmark
unzip datasets.zip
```

**4. Set your openai api key:**
```
export OPENAI_API_KEY=""
```

<!-- **4. Test your installation**:
Run a sample evaluation script to ensure everything works:

```
python evaluation/unit_test_evaluation.py \
    --output_repository_path outputs/noagent/sample_repo \
    --unit_test_evaluation_path results/unit_test_evaluation/sample_result
``` -->



##  Generation of Baseline Results

###  OpenHands

#### Environment and LLM Setup

Please follow the official setup instructions in the [OpenHands repository](https://github.com/All-Hands-AI/OpenHands/blob/main/evaluation/README.md#setup) to configure your local development environment and LLM. We use version 0.33.0 in our experiments. One way to download OpenHands repo is:
```
git clone https://github.com/All-Hands-AI/OpenHands.git
cd OpenHands
git checkout 0.33.0
```

Make sure you set the models in config.toml.

#### Preparing LMR-Bench for OpenHands

> **Integration steps:**  
> 1. **Copy code:**  
>    - Copy the folder `LMR-Bench/generation/OpenHands/evaluation/benchmarks/lmr_bench` into the corresponding path inside your OpenHands repository.
> 2. **Copy benchmark data:**  
>    - Copy the folder `datasets/` downloaded before under `OpenHands/evaluation/benchmarks/lmr_bench/benchmark/` for generation.
>
> Your final structure in OpenHands should look like:
> ```
> OpenHands/
> └── evaluation/
>     └── benchmarks/
>         └── lmr_bench/
>             └── benchmark
>                 └── datasets
>                     ├── project_folder_1/
>             └── <code and scripts>
> ```

#### Run Inference on LMR-Bench

```bash
cd OpenHands

./evaluation/benchmarks/lmrbench/scripts/run_infer.sh \
    [MODEL_CONFIG] [GIT_VERSION] [AGENT] [EVAL_LIMIT] [NUM_WORKERS] [EVAL_OUTPUT_DIR] [CACHE_PATH] [DEST_PATH]
```

- **MODEL_CONFIG:** LLM model configuration file
- **GIT_VERSION:** e.g. `HEAD`, is the git commit hash of the OpenHands version you would like to evaluate. It could also be a release tag like `0.6.2`.
- **AGENT:** e.g. `CodeActAgent`, is the name of the agent for benchmarks, defaulting to `CodeActAgent`.
- **EVAL_LIMIT:** e.g. `10`, limits the evaluation to the first `eval_limit` instances. By default, the script evaluates the entire SWE-bench_Lite test set (300 issues). Note: in order to use `eval_limit`, you must also set `agent`.
- **EVAL_OUTPUT_DIR:** Path to store OpenHands Agent's generation logs
- **CACHE_PATH:** Path for OpenHands agent’s events and cache (this can be the same as EVAL_OUTPUT_DIR)
- **DEST_PATH:** Path to store the repositories after OpenHands Agent's revision



##### Example
```
./evaluation/benchmarks/lmr_bench/scripts/run_infer.sh \
    llm.eval_gpt4o "" "" "" "" [EVAL_OUTPUT_DIR] [CACHE_PATH] [DEST_PATH]
```


<!-- The output_path above only saves logs of the agent. To save repositories revised by OpenHands, we need to revise line 68 and 162 in run_infer.py.
The revised repositories will be saved in the folder written in line 162. -->


### No Agent

To run code generation without an agent, use the following command:

```bash
sh scripts/no_agent_generation.sh [DATA_FOLDER] [DEST_PATH] [MODEL]
```
- **DATA_FOLDER:** Path to your dataset folder (e.g., benchmark/)
- **DEST_PATH:** Directory to store the generated repositories
- **MODEL:** Model used to generate code.


##  Evaluation

### Unit Test Evaluation

To run unit test evaluation on the generated repositories:

```bash
sh scripts/unit_test_evaluation.sh [DEST_PATH] [EVAL_RESULT_PATH]
```
- **DEST_PATH**: Path to the generated repositories you want to evaluate
- **EVAL_RESULT_PATH**: Directory to store unit test evaluation results
 
<!-- example:
```
sh scripts/base_agent_generation.sh /home/sxy240002/research_agent/NLPBench/benchmark/datasets_final /home/sxy240002/research_agent/NLPBench/outputs/BaseAgent/gpt4o
``` -->

Hint: If there are issues accessing Hugging Face models or datasets during evaluation, you can add your Hugging Face token to the shell script.

### LLM-as-a-Judge Evaluation
To run LLM-as-a-judge evaluation on the generated repositories:

```
sh scripts/llm_as_a_judge_evaluation.sh [DEST_PATH] [EVAL_RESULT_PATH]
```

<!-- example:
```
sh scripts/llm_as_a_judge_evaluation.sh /home/sxy240002/research_agent/NLPBench/outputs/BaseAgent/gpt4o /home/sxy240002/research_agent/NLPAgentBench/llm_as_a_judge_evaluation_results/BaseAgent/gpt4o
``` -->
- **DEST_PATH**: Path to the generated repositories you want to evaluate
- **EVAL_RESULT_PATH**: Directory to store LLM-as-a-judge evaluation results


<!-- ## 📊 Analysis

###  Data Contamination

- For each sample, performance results are saved in the `results/unit_test_evaluation/` and `results/llm_as_a_judge_evaluation/` directories.
- To assess data contamination, we compute a similarity score following the MLE-Bench approach:  
  The model’s familiarity with a document is measured as the **mean probability assigned to each token**, conditioned on all preceding tokens.
- Familiarity is calculated using the `info.json` file (the goal file) for each benchmark task. -->



##  Contributions
We would love to hear from the broader NLP, Machine Learning, and Software Engineering research communities, and we welcome any contributions, pull requests, or issues!
To do so, please either file a new pull request or issue and fill in the corresponding templates accordingly. We'll be sure to follow up shortly!

Contact person: [Zimu Wang] and [Ruochen Li] Email: zimu.wang@utdallas.edu, ruochen.li@utdallas.edu).

## Citation & license
MIT license. Check `LICENSE.md`.

If you find our work helpful, please use the following citations.

```bibtex
@misc{yan2025lmrbenchevaluatingllmagents,
      title={LMR-BENCH: Evaluating LLM Agent's Ability on Reproducing Language Modeling Research}, 
      author={Shuo Yan and Ruochen Li and Ziming Luo and Zimu Wang and Daoyang Li and Liqiang Jing and Kaiyu He and Peilin Wu and George Michalopoulos and Yue Zhang and Ziyang Zhang and Mian Zhang and Zhiyu Chen and Xinya Du},
      year={2025},
      eprint={2506.17335},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2506.17335}, 
}
```

## Related projects

<div align="center">
  <a href="https://github.com/du-nlp-lab/MLR-Copilot"><img src="-" alt="mlr-copilot" height="120px"></a>
   &nbsp;&nbsp;
</div>

## Acknowledge
We refer to the implementation of [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench)
