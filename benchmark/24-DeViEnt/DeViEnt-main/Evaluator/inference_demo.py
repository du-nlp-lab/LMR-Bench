import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BertTokenizer
from visual_text_training import BERTModelModule, set_seed


def demo_infer(hypothesis, image_path, updates,
               model_path='evaluator_weights.pth',
               gpu=7, use_classification_head=True):
    # Set seed for reproducibility
    set_seed(42)

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    model_name = "bert-large-uncased"
    model = BERTModelModule(model_name=model_name, use_classification_head=use_classification_head).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Process the image
    image_transform = Compose([
        Resize((224, 224)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image).unsqueeze(0).to(device)

    # Process the hypothesis and updates as tensors
    hypothesis_inputs = tokenizer(hypothesis, return_tensors="pt", padding=True, truncation=True).to(device)
    update_inputs = [tokenizer(update, return_tensors="pt", padding=True, truncation=True).to(device) for update in
                     updates]

    # Run inference and collect scores
    scores = []
    with torch.no_grad():
        for update_input in update_inputs:
            # Forward pass with separate tensors
            _, score = model(image, update_input['input_ids'], update_input['attention_mask'])
            scores.append(score.item())

    return scores


if __name__ == "__main__":

    hypothesis = "A dog chases a rabbit."
    image_path = "/Data/flickr30k_images/3486831913.jpg"
    updates = [
        "A rabbit could photobomb this chase, and the dog would not even look up — it’s got its eye on nothing else but its ball.",
        "With a ball tossed by its owner, the dog’s attention is fully absorbed in the game, showing zero interest in rabbits.",
        "The dog is too absorbed in chasing the ball to even notice a rabbit.",
        "The dog looks like it’s going to chase something any second now.",
        "Every muscle in the dog’s body is alert, signaling it’s primed for a chase.",
        "With that intense look, could the dog be any more ready to chase?"
    ]

    # Call the demo_infer function
    scores = demo_infer(hypothesis, image_path, updates)

    # Print the output scores for each update
    for i, score in enumerate(scores):
        print(f"Update {i + 1} score: {score}")
