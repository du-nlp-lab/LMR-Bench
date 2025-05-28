#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_and_save_pdf.py

This script extracts text from each page of a PDF file using PyMuPDF (fitz)
and saves the result as a JSON file in the input directory, with the filename 'paper.json'.
"""

import sys
import json
import fitz  # PyMuPDF
import os


def extract_pdf_text(pdf_path: str) -> dict[int, str]:
    """
    从给定的 PDF 文件中提取每一页的文本。

    参数:
        pdf_path (str): PDF 文件的路径。

    返回:
        dict[int, str]: 一个字典，键是页码（1-based），值是对应页的文本内容。
    """
    doc = fitz.open(pdf_path)
    text_by_page: dict[int, str] = {}

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text = page.get_text("text")
        # 将页码从 1 开始
        text_by_page[page_index + 1] = text

    doc.close()
    return text_by_page


def save_dict_as_json(data: dict[int, str], output_path: str) -> None:
    """
    将字典写入 JSON 文件。

    参数:
        data (dict[int, str]): 要保存的字典，键为页码，值为对应页文本。
        output_path (str): 输出 JSON 文件路径。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,  # 保留非 ASCII 字符
            indent=2             # 美化缩进
        )


def main():
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <输入 PDF 路径>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # 获取输入 PDF 所在目录，并生成 paper.json 路径
    input_dir = os.path.dirname(pdf_path)
    json_path = os.path.join(input_dir, "paper.json")

    print(f"正在从 PDF 提取文本: {pdf_path}")
    pages = extract_pdf_text(pdf_path)

    print(f"提取完成，共 {len(pages)} 页。正在保存为 JSON: {json_path}")
    save_dict_as_json(pages, json_path)

    print("已完成。")


if __name__ == "__main__":
    main()