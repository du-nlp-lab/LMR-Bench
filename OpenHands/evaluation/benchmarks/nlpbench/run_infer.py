import asyncio
import os
from typing import Any

import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import pathlib, shutil

from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    codeact_user_response,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AppConfig,
    get_llm_config_arg,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}

LOCAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'benchmark')






def format_task_dict(example):
    task = {
        'instance_id': example['instance_id'],
        'paper_name': example['paper_name'],
        'folder_name': example['folder_name'],
        'paper_url': example['paper_url'],
        'repo_url': example['repo_url'],
        'repo_folder_name': example['repo_folder_name'],
        'implementations': example['implementations'],
    }
    return task


def get_config(
    metadata: EvalMetadata,
    instance_id: str,
) -> AppConfig:
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = (
        'docker.io/xingyaoww/openhands-eval-scienceagentbench'
    )
    cache_path = "/home/sxy240002/research_agent/OpenHands/evaluation/benchmarks/nlpbench/outputs/gpt4.1"
    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime=os.environ.get('RUNTIME', 'docker'),
        max_budget_per_task=4,
        max_iterations=metadata.max_iterations,
        sandbox=sandbox_config,
        # mount workspace
        # workspace_base=os.path.join(LOCAL_DATASET_PATH, "datasets"),
        # workspace_mount_path="/workspace/benchmark/datasets",
        # workspace_mount_path_in_sandbox="/workspace/benchmark/datasets",
        workspace_base=None,
        workspace_mount_path=None,
        file_store_path=cache_path + "/file_store",
        cache_dir="/home/sxy240002/research_agent/OpenHands/evaluation/benchmarks/nlpbench/cache",
        save_trajectory_path= cache_path + "/save_trajectory/"
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config,
            metadata.eval_output_dir,
            instance_id,
        )
    )
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Initialization Fn {'-' * 50}")
    obs: CmdOutputObservation

    # # Set up workspace directories
    # action = CmdRunAction(command='mkdir -p /workspace/pred_programs')
    # logger.info(action, extra={'msg_type': 'ACTION'})
    # obs = runtime.run_action(action)
    # assert obs.exit_code == 0

    # action = CmdRunAction(command='mkdir -p /workspace/pred_results')
    # logger.info(action, extra={'msg_type': 'ACTION'})
    # obs = runtime.run_action(action)
    # assert obs.exit_code == 0

    dataset_name = instance['folder_name']

    # Copy the dataset to the workspace
    dataset_dir = os.path.join(
        LOCAL_DATASET_PATH,
        'datasets',
        dataset_name,
    )
    runtime.copy_to(dataset_dir, '/workspace/benchmark/datasets', recursive=True)

    # Check the dataset exists
    action = CmdRunAction(command='cd /workspace/benchmark/datasets && ls')
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    assert dataset_name in obs.content

    logger.info(f"{'-' * 50} END Runtime Initialization Fn {'-' * 50}")


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Completion Fn {'-' * 50}")
    obs: CmdOutputObservation

    test_result = {}

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)

    assert obs.exit_code == 0


    zip_path = runtime.copy_from("/workspace/benchmark/datasets")

    dest = pathlib.Path("/home/sxy240002/research_agent/NLPAgentBench/outputs/OpenHands/gpt4.1")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_path, dest)

    os.remove(zip_path)

    logger.info(f"{'-'*50} END Runtime Completion Fn {'-'*50}")

    # action = CmdRunAction(command=f'cat pred_programs/{instance.pred_program_name}')
    # logger.info(action, extra={'msg_type': 'ACTION'})
    # obs = runtime.run_action(action)

    # if obs.exit_code == 0:
    #     test_result = {'program': obs.content}
    # else:
    #     test_result = {'program': 'ERROR'}

    logger.info(f"{'-' * 50} END Runtime Completion Fn {'-' * 50}")
    return test_result


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    instance_id = instance.instance_id
    config = get_config(metadata, instance_id)

    # Set up the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance_id}.')

    implementations = instance.implementations
    instructions = []
    for implementation in implementations:
        instructions.append(implementation['instruction'])

    instruction_str = "\n".join([f"{index}: {element}" for index, element in enumerate(instructions)])

    instruction_1 = f"""You are an expert NLP research scientist to help user with coding on NLP related tasks. 
    You are given a research paper named paper.pdf and an incomplete repository of the paper. You need to parse the paper with Python first and then are required to finish the user's task based on the paper and the repository.
    If you cannot read the paper directly, you can try to write python code to parse the pdf file.
    
    The user's tasks include the following:"""
    instruction_2 = """ 
    You are required to directly revise the python file mentioned in the instruction. You can only revise the function you need to implement. Do not revise unrelated part of the file.

    Please do NOT run the program in the background.
    If the program uses some packages that are incompatible, please figure out alternative implementations and do NOT restart the environment.

    """

    instruction = instruction_1 + instruction_str + instruction_2


    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)
    initialize_runtime(runtime, instance)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = asyncio.run(
        run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
            fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                metadata.agent_class
            ),
        )
    )

    # ======= Attempt to evaluate the agent's edits =======
    test_result = complete_runtime(runtime, instance)

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')
    metrics = state.metrics.get() if state.metrics else None

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = compatibility_for_eval_history_pairs(state.history)

    # Save the output
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result=test_result,
    )
    return output


if __name__ == '__main__':
    parser = get_parser()
    args, _ = parser.parse_known_args()


    dataset = load_dataset("Shinyy/NLPBench", split="train", download_mode="force_redownload")

    dataset_processed = []
    for example in tqdm(dataset):
        dataset_processed.append(
            format_task_dict(example)
        )

    dataset = pd.DataFrame(dataset_processed)

    llm_config = None
    if args.llm_config:
        # print("args.llmconfig:", args.llm_config)
        llm_config = get_llm_config_arg(args.llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    metadata = make_metadata(
        llm_config,
        'NLPBench',
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    dataset['instance_id'] = dataset['instance_id'].apply(str)
    instances = prepare_dataset(dataset, output_file, args.eval_n_limit)

    run_evaluation(
        instances, metadata, output_file, args.eval_num_workers, process_instance
    )
