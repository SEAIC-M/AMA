# agent_worker.py

import pandas as pd
import argparse
from pyopenagi.agents.agent_factory import AgentFactory
from pyopenagi.agents.agent_process import AgentProcessFactory
from aios.scheduler.fifo_scheduler import FIFOScheduler
from aios.llm_core import llms

def run_agent_worker(agent_path, task_input, args_dict, tool_dict, vector_db_path, memory_dict, steal_params):

    args = argparse.Namespace(**args_dict)
    tool = pd.Series(tool_dict)

    llm = llms.LLMKernel(
        llm_name=args.llm_name,
        max_gpu_memory=args.max_gpu_memory,
        eval_device=args.eval_device,
        max_new_tokens=args.max_new_tokens,
        log_mode=args.llm_kernel_log_mode,
        use_backend=args.use_backend
    )

    scheduler = FIFOScheduler(llm=llm, log_mode=args.scheduler_log_mode)
    scheduler.start()

    agent_factory = AgentFactory(
        agent_process_queue=scheduler.agent_process_queue,
        agent_process_factory=AgentProcessFactory(),
        agent_log_mode=args.agent_log_mode,
    )

    vector_db = None

    result = agent_factory.run_agent(
        agent_name=agent_path,
        task_input=task_input,
        args=args,
        attacker_tool=tool,
        vector_db=vector_db,
        agg=tool["Aggressive"],
        memory=memory_dict,
        to_steal_params=steal_params
    )

    scheduler.stop()
    return result
