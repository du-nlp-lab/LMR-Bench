#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_and_save_pdf.py

This script extracts text from each page of a PDF file using PyMuPDF (fitz)
and saves the result as a JSON file in the input directory named 'paper.json'.
"""

import sys
import json
import fitz  # PyMuPDF
import os


def extract_pdf_text(pdf_path: str) -> dict[int, str]:
    """
    Extracts text from each page of the given PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict[int, str]: A dictionary mapping page numbers (1-based) to the text content of each page.
    """
    document = fitz.open(pdf_path)
    text_by_page: dict[int, str] = {}

    for page_index in range(len(document)):
        page = document.load_page(page_index)
        text = page.get_text("text")
        # Use 1-based page numbering
        text_by_page[page_index + 1] = text

    document.close()
    return text_by_page


def save_dict_as_json(data: dict[int, str], output_path: str) -> None:
    """
    Writes the given dictionary to a JSON file.

    Args:
        data (dict[int, str]): Dictionary with page numbers as keys and page text as values.
        output_path (str): Path where the JSON file will be saved.
    """
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(
            data,
            json_file,
            ensure_ascii=False,  # Preserve non-ASCII characters
            indent=2             # Pretty-print with 2-space indentation
        )


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Determine directory of input PDF and set output JSON path
    input_dir = os.path.dirname(pdf_path)
    json_path = os.path.join(input_dir, "paper.json")

    print(f"Extracting text from PDF: {pdf_path}")
    pages = extract_pdf_text(pdf_path)

    print(f"Extraction complete: {len(pages)} pages found. Saving to JSON: {json_path}")
    save_dict_as_json(pages, json_path)

    print("Done.")


if __name__ == "__main__":
    main()