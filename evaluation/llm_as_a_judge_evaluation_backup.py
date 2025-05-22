#!/usr/bin/env python3
import argparse
import sys
import json
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def main():
    parser = argparse.ArgumentParser(
        description="Batch run evaluations, query ChatGPT for replacement analysis, and summarize results"
    )
    parser.add_argument(
        '--evaluation_dir', required=True,
        help="Root directory containing evaluation subfolders"
    )
    parser.add_argument(
        '--evaluation_output_dir', required=True,
        help="Directory where test results and ChatGPT analyses will be saved"
    )
    args = parser.parse_args()

    eval_root = Path(args.evaluation_dir)
    output_root = Path(args.evaluation_output_dir)

    if not eval_root.is_dir():
        sys.exit(f"Error: evaluation directory not found: {eval_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    # Initialize counters for three categories
    category_counts = {'1': 0, '2': 0, '3': 0}
    category_cases = {'1': [], '2': [], '3': []}
    total = 0

    for subfolder in sorted(eval_root.iterdir()):
        if not subfolder.is_dir():
            continue
        info_path = subfolder / 'info.json'
        if not info_path.is_file():
            print(f"[WARNING] Skipping {subfolder.name}: info.json not found")
            continue

        info = json.loads(info_path.read_text(encoding='utf-8'))
        implementations = info.get('implementations', [])
        repo_name = info.get('repo_folder_name', '')

        for idx, impl in enumerate(implementations):
            output_filename = f"{subfolder.name}_{repo_name}_{idx}_analysis.txt"
            output_path = output_root / output_filename
            if output_path.exists():
                print(f"[INFO] Skipping {subfolder.name}, idx {idx}: {output_filename} already exists")
                continue
            
            instruction = impl.get('instruction', '').strip()
            goal_file = impl.get('goal_file', '')
            golden_file = impl.get('golden_file', '')

            goal_path = subfolder / repo_name / goal_file
            golden_path = subfolder / golden_file

            try:
                goal_content = goal_path.read_text(encoding='utf-8')
                golden_content = golden_path.read_text(encoding='utf-8')
            except Exception as e:
                print(f"[ERROR] Failed to read files for {subfolder.name}, idx {idx}: {e}")
                continue

            # Construct prompt with output format constraint
            prompt = (
                f"Instruction: {instruction}\n\n"
                "You are an expert NLP software engineer tasked with evaluating the correctness of a function implementation by comparing two code artifacts:\n"
                f"- Golden Reference ({golden_file}):\n```python\n{golden_content}\n```\n"
                f"- Agent Implementation ({goal_file}):\n```python\n{goal_content}\n```\n\n"
                "Instructions:\n"
                "1. Examine both implementations in detail, focusing on:\n"
                "   - Logical correctness relative to the specification provided above.\n"
                "   - Handling of edge cases and error conditions.\n"
                "   - Subtle deviations (e.g., off-by-one errors, missing checks).\n"
                "2. Classify your judgment into exactly one of the following categories:\n"
                "   1. Incorrect Logic: The core algorithm deviates from the specification and produces wrong results.\n"
                "   2. Logic Correct but Subtle Errors: The main algorithm matches the specification, but there are other implementation mistakes or omissions.\n"
                "   3. Completely Correct: The implementation is fully faithful to the specification with no errors.\n"
                "3. For your chosen category, provide a concise rationale (2–4 bullet points) illustrating the key discrepancies or confirmations.\n\n"
                "Output Format (JSON):\n"
                "{\n"
                '  "category": "<1 | 2 | 3>",\n'
                '  "rationale": [\n'
                '    "First key point…",\n'
                '    "Second key point…"\n'
                '  ]\n'
                "}\n"
)




            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-2025-04-14",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[ERROR] OpenAI API call failed for {subfolder.name}, idx {idx}: {e}")
                continue

            # Save raw LLM output
            output_filename = f"{subfolder.name}_{repo_name}_{idx}_analysis.txt"
            output_path = output_root / output_filename
            output_path.write_text(answer, encoding='utf-8')
            print(f"[INFO] Saved analysis to {output_path}")

            # ---- NEW: parse JSON response ----
            try:
                result = json.loads(answer)
                category = str(result["category"]).strip()
                rationale = result.get("rationale", [])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARNING] Failed to parse JSON for {subfolder.name}, idx {idx}: {e}")
                continue

            if category in category_counts:
                category_counts[category] += 1
                category_cases[category].append({
                    "subfolder": subfolder.name,
                    "repo": repo_name,
                    "index": idx,
                    "rationale": rationale
                })
                total += 1
            else:
                print(f"[WARNING] Unexpected category '{category}' in response for {subfolder.name}, idx {idx}")

        # ---- rewrite your summary to include counts only (rationales saved in category_cases) ----
        summary_path = output_root / 'summary.txt'
        with summary_path.open('w', encoding='utf-8') as summary_file:
            summary_file.write(f"Total cases: {total}\n")
            for cat in ['1', '2', '3']:
                count = category_counts[cat]
                pct   = count / total * 100 if total > 0 else 0
                summary_file.write(f"Category {cat} count: {count} ({pct:.2f}%)\n")
            summary_file.write("\nDetailed cases by category:\n")
            for cat in ['1', '2', '3']:
                cases = category_cases[cat]
                if not cases:
                    continue
                summary_file.write(f"\nCategory {cat}:\n")
                for c in cases:
                    summary_file.write(
                        f"- Subfolder: {c['subfolder']}, Repo: {c['repo']}, Index: {c['index']}\n"
                    )


if __name__ == '__main__':
    main()
