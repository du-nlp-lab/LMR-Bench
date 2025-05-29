from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_LAYERS_MAPPINGS = {
    "Phi-1_5": "model.model.layers",
    "Phi-1": "model.model.layers",
    "Phi-2": "model.model.layers",
    "Mistral-7B-v0.1": "model.model.layers",
    "Meta-Llama-3-8B": "model.model.layers",
    "EleutherAI_pythia-1.4b": "model.gpt_neox.layers",
}

MODULE_MAPPINGS = {
    "mlp": {
        "Phi-1_5": "mlp.fc2",
        "Mistral-7B-v0.1": "mlp.down_proj",
        "EleutherAI_pythia-1.4b": "mlp.dense_4h_to_h",
        "Meta-Llama-3-8B": "mlp.down_proj",
    },
    "mlp-l1": {
        "Phi-1_5": "mlp.activation_fn",
        "Mistral-7B-v0.1": "mlp.act_fn",
        "EleutherAI_pythia-1.4b": "mlp.act",
        "Meta-Llama-3-8B": "mlp.act_fn",
    },
    "mhsa": {
        "Phi-1_5": "self_attn.dense",
        "Mistral-7B-v0.1": "self_attn.o_proj",
        "EleutherAI_pythia-1.4b": "attention.dense",
        "Meta-Llama-3-8B": "self_attn.o_proj",
    },
}

"""
Here are the steps to get the activations in PyTorch:
    1. Register a hook function at specific module layers to store the activations
    2. Apply a forward pass in order to compute the activations
    3. Don't forget to remove the hooks
"""


def return_hook_function(
    module: str,
    first_token_idx: int,
    object_context_idx: int,
    subject_query_idx: int,
    relation_query_idx: int,
    num_layers: int,
    include_first_token: bool,
    include_object_context_token: bool,
    include_subject_query_token: bool,
    include_relation_query_token: bool,
    vertical: bool,
    activations: dict,
) -> Callable:
    """this function allows to return the hook function that gets and stores the activations for a specific module, position, and for all layers of the LLM

    Args:
        module (str): a string specifying the module (mlp, mlp-l1, or mhsa)
        first_token_idx (int): the position (index) of the first token
        object_context_idx (int): the position (index) of the object
        subject_query_idx (int): the position (index) of the subject in the query
        relation_query_idx (int): the position (index) of the relation in the query
        num_layers (int): the number of transformers blocks (or layers)
        include_first_token (bool): whether to probing on the first token or not (see control experiment of the paper)
        include_object_context_token (bool): whether to probing on the object token in the context or not
        include_subject_query_token (bool): whether to probing on the object token in the query or not
        include_relation_query_token (bool): whether to probing on the relation token in the query or not
        vertical (bool): whether to store the activations vertically (i.e. by token) or no
        activations (dict): a dictionary to store the activations

    Returns:
        function: renders a hook function to get the activations of MLPs and MHSAs
    """

    def hook(model: AutoModelForCausalLM, act_input: torch.Tensor, act_output: torch.Tensor) -> None:
        """the hook function with the pytorch specific hook signature to process a forward pass's input/output for a model (LLM)

        Args:
            model (AutoModelForCausalLM): the auto-regressive LLM
            input (torch.Tensor): the input of the hooked function/module
            act_output (torch.Tensor): the output/activations of the hooked function/module
        """
        token_positions = {
            "object": object_context_idx,
            "subject_query": subject_query_idx,
            "relation_query": relation_query_idx,
            "first": first_token_idx,
        }
        if vertical:  # when vertical is True, the position specifies the token on which to focus
            positions = []
            if include_object_context_token:
                positions.append("object")
            if include_subject_query_token:
                positions.append("subject_query")
            if include_relation_query_token:
                positions.append("relation_query")
            if include_first_token:
                positions.append("first")

            for position in positions:
                layer_activations = act_output[0, token_positions[position]].to("cpu").numpy()

                if len(activations[module]) == 0 or len(activations[module][-1][position]) == num_layers:
                    activations[module].append({"first": [], "object": [], "subject_query": [], "relation_query": []})

                activations[module][-1][position].append([layer_activations])

        else:  # when vertical is False, we take all the positions
            layer_activations_object = act_output[0, object_context_idx].to("cpu")
            layer_activations_subject = act_output[0, subject_query_idx].to("cpu")
            layer_activations_relation = act_output[0, relation_query_idx].to("cpu")
            layer_activations_first = act_output[0, first_token_idx].to("cpu")

            if len(activations[module]) == 0 or len(activations[module][-1]) == num_layers:
                activations[module].append([])

            activations[module][-1].append(
                [
                    layer_activations_first,
                    layer_activations_object,
                    layer_activations_subject,
                    layer_activations_relation,
                ]
            )

    return hook


def register_forward_hooks(
    model: AutoModelForCausalLM,
    model_name: str,
    num_layers: int,
    module_type: str,
    return_hook_function,
    first_token_idx: int,
    object_context_idx: int,
    subject_query_idx: int,
    relation_query_idx: int,
    include_first_token: bool,
    include_object_context_token: bool,
    include_subject_query_token: bool,
    include_relation_query_token: bool,
    vertical: bool,
    activations: dict,
    model_layers_mappings: dict = MODEL_LAYERS_MAPPINGS,
    module_mappings: dict = MODULE_MAPPINGS,
):
    """
    Registers forward hooks, generalize for modules names that are specific to each LLM

    Args:
        model (AutoModelForCausalLM): The loaded LLM.
        model_name (str): The name of the model to determine layer structure.
        num_layers (int): The number of hidden layers in the model.
        module_type (str): The type of module to register hooks for ("mlp", "mlp-l1", "mhsa").
        return_hook_function (function): The function to capture activations.
        first_token_idx (int): the position (index) of the first token
        object_context_idx (int): Index of the object token.
        subject_query_idx (int): Index of the subject token.
        relation_query_idx (int): Index of the relation token.
        include_first_token (bool): whether to probing on the first token or not (see control experiment of the paper)
        include_object_context_token (bool): whether to probing on the object token in the context or not
        include_subject_query_token (bool): whether to probing on the object token in the query or not
        include_relation_query_token (bool): whether to probing on the relation token in the query or not
        vertical (bool): Whether to get the activations by token or not (i.e. by layer)
        model_layers_mappings (dict): a dict mapping between model name and the layer structure of each specific LLM
        module_mappings (dict): a dict mapping between model name and MLPs/MHSAs structure for each specific LLM
        activations (dict): a dictionary to store all the activations

    Returns:
        list: A list of registered hooks.
    """
    # global activations

    hooks = []

    if model_name not in model_layers_mappings:
        raise ValueError("Model not specified!")

    for layer in range(num_layers):
        layer_module = eval(f"{model_layers_mappings[model_name]}[{layer}].{module_mappings[module_type][model_name]}")
        hooks.append(
            layer_module.register_forward_hook(
                return_hook_function(
                    module=module_type,
                    first_token_idx=first_token_idx,
                    object_context_idx=object_context_idx,
                    subject_query_idx=subject_query_idx,
                    relation_query_idx=relation_query_idx,
                    num_layers=num_layers,
                    include_first_token=include_first_token,
                    include_object_context_token=include_object_context_token,
                    include_subject_query_token=include_subject_query_token,
                    include_relation_query_token=include_relation_query_token,
                    vertical=vertical,
                    activations=activations,
                )
            )
        )

    return hooks


def forward_pass_with_hooks(
    model: AutoModelForCausalLM,
    model_name: str,
    tokenizer: AutoTokenizer,
    knowledge_probing_prompt: str,
    device: str,
    first_token_idx: int,
    object_context_idx: int,
    subject_query_idx: int,
    relation_query_idx: int,
    include_first_token: bool,
    include_object_context_token: bool,
    include_subject_query_token: bool,
    include_relation_query_token: bool,
    include_mlps: bool,
    include_mlps_l1: bool,
    include_mhsa: bool,
    vertical: bool,
    activations: dict,
) -> None:
    """
    Registers activation hooks and applies a forward pass for the model.

    Args:
        model (AutoModelForCausalLM): The loaded LLM.
        model_name (str): the name of the LLM
        tokenizer (AutoTokenizer): The tokenizer for the model.
        knowledge_probing_prompt (str): The input prompt for generation.
        device (str): The device to run inference on.
        first_token_idx (int): The index for the first token.
        object_context_idx (int): The index for the object token.
        subject_query_idx (int): The index for the subject query token.
        relation_query_idx (int): The index for the relation query token.
        include_first_token (bool): whether to probing on the first token or not (see control experiment of the paper)
        include_object_context_token (bool): whether to probing on the object token in the context or not
        include_subject_query_token (bool): whether to probing on the object token in the query or not
        include_relation_query_token (bool): whether to probing on the relation token in the query or not
        include_mlps (bool): Whether to include MLP activations.
        include_mlps_l1 (bool): Whether to include MLP-L1 activations.
        include_mhsa (bool): Whether to include MHSA activations.
        vertical (bool): Whether the token positions are vertical.
        activations (dict): a dictionary to store the activations
    """

    if include_mlps:
        hooks = register_forward_hooks(
            model,
            model_name,
            model.config.num_hidden_layers,
            "mlp",
            return_hook_function,
            first_token_idx,
            object_context_idx,
            subject_query_idx,
            relation_query_idx,
            include_first_token,
            include_object_context_token,
            include_subject_query_token,
            include_relation_query_token,
            vertical,
            activations,
        )

        with torch.no_grad():
            model(tokenizer(knowledge_probing_prompt, return_tensors="pt").to(device).input_ids)
        for hook in hooks:
            hook.remove()

    if include_mlps_l1:
        hooks = register_forward_hooks(
            model,
            model_name,
            model.config.num_hidden_layers,
            "mlp-l1",
            return_hook_function,
            first_token_idx,
            object_context_idx,
            subject_query_idx,
            relation_query_idx,
            include_first_token,
            include_object_context_token,
            include_subject_query_token,
            include_relation_query_token,
            vertical,
            activations,
        )
        with torch.no_grad():
            model(tokenizer(knowledge_probing_prompt, return_tensors="pt").to(device).input_ids)
        for hook in hooks:
            hook.remove()

    if include_mhsa:
        hooks = register_forward_hooks(
            model,
            model_name,
            model.config.num_hidden_layers,
            "mhsa",
            return_hook_function,
            first_token_idx,
            object_context_idx,
            subject_query_idx,
            relation_query_idx,
            include_first_token,
            include_object_context_token,
            include_subject_query_token,
            include_relation_query_token,
            vertical,
            activations,
        )
        with torch.no_grad():
            model(tokenizer(knowledge_probing_prompt, return_tensors="pt").to(device).input_ids)
        for hook in hooks:
            hook.remove()
