#!/usr/bin/env python3
import argparse
import os
import json
import shutil
import ast

# Replace OpenAI client with Anthropic client for Claude 3.7 Sonnet
from anthropic import Anthropic

class UniversalReplacer(ast.NodeTransformer):
    """
    An AST transformer that replaces either a top‐level function or a method
    in a specified class (or any class if target_class is None) with a new
    implementation returned by the LLM.
    """
    def __init__(self, target_method: str, new_code: str, target_class: str = None):
        if target_class == "":
            target_class = None
        self.target_class = target_class
        self.target_method = target_method
        self.class_stack = []
        # Parse the LLM‐returned function definition into an AST node
        self.new_node = ast.parse(new_code).body[0]

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        # Push current class name, recurse, then pop
        self.class_stack.append(node.name)
        new_node = self.generic_visit(node)
        self.class_stack.pop()
        return new_node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Determine if we're in the right class (or any if target_class is None)
        in_target_class = (
            self.target_class is None or
            (self.class_stack and self.class_stack[-1] == self.target_class)
        )
        if node.name == self.target_method and in_target_class:
            # First recurse children, then replace this node
            self.generic_visit(node)
            return self.new_node
        return self.generic_visit(node)


def replace_method_in_file(
    src_path: str,
    method_name: str,
    new_code: str,
    class_name: str = None
):
    """
    Read src_path, replace the function/method named method_name (in class_name if given)
    with new_code via AST, then overwrite the file in place.
    """
    source = open(src_path, 'r', encoding='utf-8').read()
    tree = ast.parse(source)
    transformer = UniversalReplacer(method_name, new_code, class_name)
    new_tree = transformer.visit(tree)
    new_source = ast.unparse(new_tree)
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write(new_source)


def main():
    parser = argparse.ArgumentParser(
        description="Copy each project to a new directory, then use Claude-Sonnet-3.7 + AST to replace exactly one function/method."
    )
    parser.add_argument("--input_dir",  required=True, help="Source projects directory")
    parser.add_argument("--output_dir", required=True, help="Destination directory for modified projects")
    parser.add_argument("--api_key",
                        default=os.getenv("ANTHROPIC_API_KEY", ""),
                        help="Anthropic API key for Claude-Sonnet-3.7")
    args = parser.parse_args()

    # Initialize Anthropic client for Claude 3.7 Sonnet
    client = Anthropic(api_key=args.api_key)

    for proj in os.listdir(args.input_dir):
        src_proj = os.path.join(args.input_dir, proj)
        info_path = os.path.join(src_proj, "info.json")
        if not os.path.isdir(src_proj) or not os.path.isfile(info_path):
            continue

        dst_proj = os.path.join(args.output_dir, proj)
        # Skip if already processed
        if os.path.isdir(dst_proj):
            print(f"[{proj}] skipped")
            continue

        # 1) Copy entire project directory
        shutil.copytree(src_proj, dst_proj, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('.git', '*.idx', '*.pack'))

        # 2) Load metadata and paper JSON
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        repo_folder = os.path.join(dst_proj, info["repo_folder_name"])
        paper_path  = os.path.join(repo_folder, "paper.json")
        paper_json  = json.load(open(paper_path, 'r', encoding='utf-8'))

        # 3) Process each implementation task
        for impl in info.get("implementations", []):
            goal_rel    = impl["goal_file"]
            method_name = impl['goal_function']
            class_name  = impl.get('class_name')
            file_path   = os.path.join(repo_folder, goal_rel)

            # Read full source for context
            with open(file_path, 'r', encoding='utf-8') as f:
                full_src = f.read()

            # Build prompt: full file + paper + instruction + related code
            prompt_parts = [
                "You are a code assistant.",
                "Below is the **entire** Python source file.",
                "Please implement **only** the function/method named ``" + method_name + "``.",
                "Return **only** its def line and indented body—no fences or explanations.",
                "", "=== FILE BEGIN ===", full_src.rstrip(), "=== FILE END ===", "",
                "Paper (JSON):", json.dumps(paper_json, indent=2), "",
                "Instruction:", impl["instruction"], ""
            ]
            if impl.get("retrieval_content", []):
                prompt_parts.append("Related code for reference:")
                for rc in impl["retrieval_content"]:
                    rc_path = os.path.join(repo_folder, rc)
                    if os.path.isfile(rc_path):
                        prompt_parts.append(f"# Path: {rc}")
                        prompt_parts.append(open(rc_path, 'r', encoding='utf-8').read())
                prompt_parts.append("")

            prompt = "\n".join(prompt_parts)

            # Call Claude-Sonnet-3.7 to get only the new function/method
            resp = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096
            )
            blocks = resp.content
            text = "".join([blk["text"] for blk in blocks])
            new_code = text.strip()

            # 4) Replace via AST and write back
            replace_method_in_file(file_path, method_name, new_code, class_name)
            print(f"[{proj}] ✓ {goal_rel} has been replaced")

if __name__ == "__main__":
    main()
