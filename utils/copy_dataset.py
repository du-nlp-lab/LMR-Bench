import os
import shutil
import sys
def copy_and_rename_outputs(src_root):
    """
    复制 src_root/outputs 下的所有内容，
    在同一目录下创建 outputs_renamed 并将内容拷贝过去。
    """
    # 定义源目录和目标目录
    src_dir = os.path.join(src_root, "outputs")
    dst_dir = os.path.join(src_root, "outputs_renamed")
 
    # 检查源目录是否存在
    if not os.path.isdir(src_dir):
        print(f"错误：源目录不存在 → {src_dir}", file=sys.stderr)
        sys.exit(1)
 
    # 确保目标目录不存在，防止覆盖
    if os.path.exists(dst_dir):
        print(f"错误：目标目录已存在 → {dst_dir}", file=sys.stderr)
        sys.exit(1)
 
    try:
        # 递归拷贝目录
        shutil.copytree(src_dir, dst_dir)
        print(f"复制成功：{src_dir} → {dst_dir}")
    except Exception as e:
        print(f"复制失败：{e}", file=sys.stderr)
        sys.exit(1)
root = "/home/sxy240002/research_agent/OpenHands/evaluation/benchmarks/nlpbench"
copy_and_rename_outputs(root)