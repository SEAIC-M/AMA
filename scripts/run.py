import os
import yaml
import argparse
import datetime
from multiprocessing import Process, Queue
from tqdm import tqdm
import re
def run_llm_task(llm, cfg, task_queue: Queue):
    attack_tool_types = cfg.get('attack_tool', [])
    write_db = cfg.get('write_db', None)
    read_db = cfg.get('read_db', None)
    injection_method = cfg.get('injection_method', None)
    skip_if_exists = cfg.get('skip_if_exists', True)
    task_num = cfg.get('task_num', 5)
    parallel_num = cfg.get('parallel_num', 1)
    each_run_num = cfg.get('each_run_num', 1)
    attack_types_cfg = cfg.get('attack_types', {})
    run_tools_num = cfg.get('run_tools_num', 2)
    steal_privacy_type = cfg.get('steal_privacy_type', None)
    suffix = ""
    suffix += f"_{steal_privacy_type}" if steal_privacy_type else ""

    for i in range(each_run_num):
        for attack_type, type_cfg in attack_types_cfg.items():
            steal_parmas_nums = type_cfg.get("steal_parmas_num", [])
            defense_types = type_cfg.get("defense_type", []) or [None]

            for steal_parmas_num in steal_parmas_nums:
                for defense_type in defense_types:
                    llm_name = llm
                    backend = None

                    for attack_tool_type in attack_tool_types:

                        if attack_tool_type == 'untarget':
                            attacker_tools_path = 'data/generated_untarget_attack.jsonl'
                        elif attack_tool_type == 'target':
                            attacker_tools_path = 'data/generated_target_attack_tools.jsonl'
                        elif attack_tool_type == 'original':
                            attacker_tools_path = 'data/all_attack_tools.jsonl'


                        log_path = f'logs/{attack_tool_type}/{llm_name}/attack/{attack_type}/steal_parmas_num_{steal_parmas_num}'
                        if defense_type:
                            log_path = f'logs/{attack_tool_type}/{llm_name}/defense/{defense_type}/attack/{attack_type}/steal_parmas_num_{steal_parmas_num}'

                        current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
                        log_file = f'{log_path}/{current_date}{suffix}.xlsx'
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)

                        base_cmd = f'''python -u main_attacker.py --llm_name {llm} --attack_type {attack_type} --use_backend {backend} --attacker_tools_path {attacker_tools_path} --res_file {log_file}'''
                        base_cmd += f' --run_tools_num {run_tools_num} --steal_parmas_num {steal_parmas_num} --task_num {task_num} --parallel_num {parallel_num} --attack_tool_type {attack_tool_type}'
                        
                        if write_db:
                            base_cmd += ' --write_db'
                        if read_db:
                            base_cmd += ' --read_db'
                        if defense_type:
                            base_cmd += f' --defense_type {defense_type}'
                        
                        if steal_privacy_type:
                            base_cmd += f' --steal_privacy_type {steal_privacy_type}'

                        specific_cmd = ''
                        if injection_method in ['direct_prompt_injection', 'memory_attack', 'observation_prompt_injection', 'clean']:
                            specific_cmd = f' --{injection_method}'
                        elif injection_method == 'mixed_attack':
                            specific_cmd = ' --direct_prompt_injection --observation_prompt_injection'
                        elif injection_method == 'DPI_MP':
                            specific_cmd = ' --direct_prompt_injection'
                        elif injection_method == 'OPI_MP':
                            specific_cmd = ' --observation_prompt_injection'
                        elif injection_method == 'DPI_OPI':
                            specific_cmd = ' --direct_prompt_injection --observation_prompt_injection'

                        cmd = f"{base_cmd}{specific_cmd} > {log_file}_{suffix}.log 2>&1"

                        def remove_timestamp_prefix(filename):
                            return re.sub(r'^\d{4}-\d{2}-\d{2}-\d{2}:\d{2}', '', filename)

                        if os.path.exists(log_path):
                            existing_xlsx = []
                            for f in os.listdir(log_path):
                                if f.endswith(".xlsx"):
                                    existing_xlsx.append(remove_timestamp_prefix(f))
                        else:
                            existing_xlsx = []

                        existing_xlsx = [f for f in existing_xlsx if f==f"{suffix}.xlsx"]
                        if skip_if_exists and len(existing_xlsx) >= each_run_num:
                            task_queue.put("[Skipped]")
                            continue

                        print(f"Running command: {cmd}")
                        task_queue.put("[Started]")
                        os.system(cmd)
                        task_queue.put("[Finished]")

def main():
    parser = argparse.ArgumentParser(description='Load YAML config file')
    parser.add_argument('-c','--cfg_path',type=str, default='./config/target_new.yml', help='Path to the YAML configuration file')
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    llms = cfg.get('llms', [])
    task_queue = Queue()
    processes = []

    task_types = cfg.get('attack_types', {})
    total_tasks = sum(
        len(v.get("steal_parmas_num", [])) *  max(1, len(v.get("defense_type", [])) * max(1, len(v.get("attack_tool",[]))) ) for v in task_types.values()
    ) * len(llms) * len(task_types)

    total_tasks = total_tasks * cfg.get('each_run_num', 1)

    with tqdm(total=total_tasks, desc="LLM Task Progress") as pbar:
        for llm in llms:
            p = Process(target=run_llm_task, args=(llm, cfg, task_queue))
            p.start()
            processes.append(p)

        finished_tasks = 0
        while finished_tasks < total_tasks:
            msg = task_queue.get()
            if msg in ("[Finished]", "[Skipped]"):
                pbar.update(1)
                finished_tasks += 1

        for p in processes:
            p.join()

if __name__ == '__main__':
    main()