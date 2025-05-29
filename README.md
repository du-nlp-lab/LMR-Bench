# LMR-BENCH: Evaluating LLM Agent’s Ability on Reproducing Language Modeling Research


# Datasets & Scripts.
Repo to be revised. Coming soon.

### Abstract: 
 Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present \ours, a comprehensive benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from NLP research papers. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked methods, and instructions for implementing these methods.
We conduct extensive experiments in standalone and agent-based settings on state-of-the-art LLMs, evaluating the accuracy of unit tests and performing both LLM and human evaluation of code correctness.
Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLMs’ ability to autonomously reproduce scientific research. We will release our benchmark and code after publication.


### Environment
python $\geq$ 3.12
```
pip install -r requirements.txt
```


### Generation
#### OpenHands
cd OpenHands
./evaluation/benchmarks/lmrbench/scripts/run_infer.sh llm.eval_gpt41 "" "" "" "" "output_path"

The output_path above only saves logs of the agent. To save repositories revised by OpenHands, we need to revise line 68 and 162 in run_infer.py.
The revised repositories will be saved in the folder written in line 162.

example:
```
./evaluation/benchmarks/nlpbench/scripts/run_infer.sh llm.eval_claude35 "" "" "" "" /home/sxy240002/research_agent/OpenHands/evaluation/benchmarks/nlpbench/outputs/claude3.5
```

#### No Agent
sh scripts/base_agent_generation.sh "datasets_folder" "output_repository_path"

example:
```
sh scripts/base_agent_generation.sh /home/sxy240002/research_agent/NLPBench/benchmark/datasets_final /home/sxy240002/research_agent/NLPBench/outputs/BaseAgent/gpt4o
```

### Evaluation
#### Unit test evaluation
sh scripts/unit_test_evaluation.sh "output_repository_path" "unit_test_evaluation_path"

example:
```
sh scripts/base_agent_generation.sh /home/sxy240002/research_agent/NLPBench/benchmark/datasets_final /home/sxy240002/research_agent/NLPBench/outputs/BaseAgent/gpt4o
```

#### LLM-as-a-judge evaluation
sh scripts/llm_as_a_judge_evaluation.sh "output_repository_path" "unit_test_evaluation_path"

example:
```
sh scripts/llm_as_a_judge_evaluation.sh /home/sxy240002/research_agent/NLPBench/outputs/BaseAgent/gpt4o /home/sxy240002/research_agent/NLPAgentBench/llm_as_a_judge_evaluation_results/BaseAgent/gpt4o
```