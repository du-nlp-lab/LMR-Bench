import json
import os 
import sys
model_names = ['gpt4.1', 'o4mini', 'gpt4o']
uuid_paper_name_mapping_dict = {}
for model_name in model_names:
    cur_root_path = f'/home/sxy240002/research_agent/OpenHands/evaluation/benchmarks/nlpbench/outputs/{model_name}/file_store/sessions'
    for paper_uuid in os.listdir(cur_root_path):
        session_cache_path = os.path.join(cur_root_path, paper_uuid, 'event_cache')
        if not os.path.isdir(session_cache_path):
            continue
        all_session_files = os.listdir(session_cache_path)
        for session_name in all_session_files:
            cur_file_name = os.path.join(session_cache_path,session_name)
            with open(cur_file_name, 'r') as f:
                cur_session_data = json.load(f)
            for step in cur_session_data:
                if 'action' in step:
                    if step['action'] == 'read':
                        path_list = step['args']['path'].split('/')
                        if len(path_list) > 4:
                            if path_list[3] == 'datasets':
                                paper_name = path_list[4]
                                if (model_name, paper_uuid) not in uuid_paper_name_mapping_dict:
                                    uuid_paper_name_mapping_dict.update({(model_name, paper_uuid):paper_name})
                                continue
 
import os
import sys
 
# 根目录
root_path = '/home/sxy240002/research_agent/NLPAgentBench/OpenHands_logs'
 
for model_name in model_names:
    sessions_dir = os.path.join(root_path, model_name, 'file_store', 'sessions')
    if not os.path.isdir(sessions_dir):
        print(f"跳过：未找到目录 {sessions_dir}", file=sys.stderr)
        continue
 
    for old_uuid in os.listdir(sessions_dir):
        old_path = os.path.join(sessions_dir, old_uuid)
 
        # 只处理目录，跳过文件
        if not os.path.isdir(old_path):
            continue
 
        # 从映射表中获取新的文件夹名
        key = (model_name, old_uuid)
        if key not in uuid_paper_name_mapping_dict:
            print(f"警告：映射中找不到 {key}", file=sys.stderr)
            continue
 
        new_name = uuid_paper_name_mapping_dict[key]
        new_path = os.path.join(sessions_dir, new_name)
 
        # 避免覆盖已存在的目录
        if os.path.exists(new_path):
            print(f"目标已存在，跳过重命名：{new_path}", file=sys.stderr)
            continue
 
        # 执行重命名
        try:
            os.rename(old_path, new_path)
            print(f"重命名成功：{old_path} → {new_path}")
        except Exception as e:
            print(f"重命名失败：{old_path} → {new_path}，错误：{e}", file=sys.stderr)