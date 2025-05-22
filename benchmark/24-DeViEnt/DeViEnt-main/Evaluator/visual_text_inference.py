import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from visual_text_training import VisualTextDataset, BERTModelModule, set_seed, process_single_df, generate_filename, \
    test
from transformers import BertTokenizer, BertModel, BertConfig, AdamW


def parse_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv_file', type=str, default='Data/DVE_test.csv',
                        help='CSV file containing image paths, captions, and targets for testing.')
    parser.add_argument('--image_dir', type=str, default='Data/flickr30k_images',
                        help='Directory containing images.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference.')
    parser.add_argument('--model_path', type=str,
                        default='evaluator_weights.pth',
                        help='Path to the fine-tuned model checkpoint.')
    parser.add_argument('--output_file', type=str, default='visual_text_bs64.csv',
                        help='File to save the inference results.')
    parser.add_argument('--gpu', type=int, default=7, help='GPU id to use.')
    parser.add_argument('--use_classification_head', action='store_true', help='Whether to use classification head.')
    parser.add_argument('--calculate_accuracy', action='store_true', help='Whether to calculate the acc')
    return parser.parse_args()


def get_score(df, batch_size=64, model_path='evaluator_weights.pth',
              gpu=7, use_classification_head=True, image_dir='Data/flickr30k_images'):
    set_seed(42)
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    test_image_paths, test_hypotheses, test_premises, test_updates, test_update_types = process_single_df(df, image_dir)

    image_transform = Compose([
        Resize((224, 224)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    model_name = "bert-large-uncased"
    model = BERTModelModule(model_name=model_name, use_classification_head=use_classification_head).to(device)

    model.load_state_dict(torch.load(model_path))

    test_dataset = VisualTextDataset(test_image_paths, test_hypotheses, test_premises, test_updates, test_update_types,
                                     tokenizer, image_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    results_df = test(model, test_loader, device, args=None, return_df=True, calculate_accuracy=False)
    return results_df


def main():
    args = parse_infer_args()
    set_seed(42)

    test_df = pd.read_csv(args.test_csv_file)

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    test_image_paths, test_hypotheses, test_premises, test_updates, test_update_types = process_single_df(test_df,
                                                                                                          args.image_dir)

    image_transform = Compose([
        Resize((224, 224)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model_name = "bert-large-uncased"
    model = BERTModelModule(model_name=model_name, use_classification_head=args.use_classification_head).to(device)

    model.load_state_dict(torch.load(args.model_path))

    test_dataset = VisualTextDataset(test_image_paths, test_hypotheses, test_premises, test_updates, test_update_types,
                                     tokenizer, image_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.calculate_accuracy:
        results_df = test(model, test_loader, device, args, return_df=True, calculate_accuracy=True)
    else:
        results_df = test(model, test_loader, device, args, return_df=True, calculate_accuracy=False)
    final_df = pd.concat([test_df, results_df], axis=1)
    final_df.to_csv(args.output_file, index=False)
    print(f"Test results saved to {args.output_file}")


if __name__ == "__main__":
    main()
