# LMR-BENCH: Evaluating LLM Agent’s Ability on Reproducing Language Modeling Research


# Datasets & Scripts.
Repo to be revised. Coming soon.

### Abstract: 
 Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present \ours, a comprehensive benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from NLP research papers. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked methods, and instructions for implementing these methods.
We conduct extensive experiments in standalone and agent-based settings on state-of-the-art LLMs, evaluating the accuracy of unit tests and performing both LLM and human evaluation of code correctness.
Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLMs’ ability to autonomously reproduce scientific research. We will release our benchmark and code after publication.


### Generation
#### OpenHands
cd OpenHands
./evaluation/benchmarks/nlpbench/scripts/run_infer.sh llm.eval_gpt41 "" "" "" "" OpenHands/evaluation/benchmarks/nlpbench/outputs/gpt4.1

#### No Agent
sh scripts/base_agent_generation.sh benchmark outputs/BaseAgent/gpt4o


### Evaluation
#### Unit test evaluation
sh scripts/unit_test_evaluation.sh outputs/BaseAgent/gpt4o unit_test_evaluation_results/BaseAgent/gpt4o

#### LLM-as-a-judge evaluation
sh scripts/llm_as_a_judge_evaluation.sh outputs/BaseAgent/gpt4o llm_as_a_judge_evaluation_results/BaseAgent/gpt4o

