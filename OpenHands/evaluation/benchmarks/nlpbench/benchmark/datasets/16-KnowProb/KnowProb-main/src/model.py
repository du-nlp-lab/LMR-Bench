from transformers import AutoModelForCausalLM, AutoTokenizer

# specify the paths to the models here if you don't want to get from the hugging face hub
MODEL_PATHS = {
    "EleutherAI_pythia-1.4b": "EleutherAI/pythia-1.4b",
    "Phi-1_5": "microsoft/phi-1_5",
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
}

# specify your huggingface access token here to load the LLMs from the modelhub
HUGGINGFACE_TOKEN = "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"


def load_model_and_tokenizer(
    model_name: str, device: str, model_paths: dict = MODEL_PATHS, huggingface_token: str=HUGGINGFACE_TOKEN
):
    """this function returns the LLM and its tokenizer that correspond to model_name

    Args:
        model_name (str): the name of the LLM
        device (str): the device where to run inference

    Returns:
        tuple: the LLM and tokenizer respectively.
    """
    print("Load model and tokenizer...")

    path = model_paths[model_name]

    model = AutoModelForCausalLM.from_pretrained(path, device_map=device, trust_remote_code=True, token=HUGGINGFACE_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(path, device_map=device, trust_remote_code=True, token=HUGGINGFACE_TOKEN)
    return model, tokenizer
