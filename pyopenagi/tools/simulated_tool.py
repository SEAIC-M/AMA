from .base import BaseTool
import pandas as pd

class SimulatedTool(BaseTool):
    def __init__(self, name, tools_info_path):
        super().__init__()
        self.name = name
        tools_info = pd.read_json(tools_info_path, lines=True)
        tool = tools_info[tools_info['Tool Name'] == name]
        self.description = tool['Description'].iloc[0]
        self.expected_achivement = tool['Expected Achievements'].iloc[0]
        self.parameters = None

    def run(self, params=None) -> str:
        print(f" NORMAL |Simulated tool {self.name} is running with params: {params}")
        return self.expected_achivement

    def get_tool_call_format(self):
        tool_call_format = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
        return tool_call_format

class AttackerTool(BaseTool):
    def __init__(self, attacker_tool):
        super().__init__()
        print(attacker_tool)
        self.tool_name=attacker_tool['Attacker Tool']
        self.tool_description=attacker_tool['Description']
        self.attack_goal = attacker_tool['Attack goal']
        parameters = attacker_tool.get('Params',None)
        self.parameters = None
        if parameters:
            self.parameters ={
                            "type": "object",
                            "properties": {
                                p['name']: {
                                    "type": "string",
                                    "description": p['description']
                                } for p in parameters
                            },
                            "required": [
                                p['name'] for p in parameters
                            ]
                }
        # self.parameters = None

    def run(self,params=None):
        print(f"Attacker tool {self.tool_name} is running with params: {params}")
        return f"You have used the attacker tool {self.tool_name}. You have achieved the attacker goal: {self.attack_goal}"

    def get_tool_call_format(self):
        tool_call_format = {
            "type": "function",
            "function": {
                "name": f"{self.tool_name}",
                "description": f"{self.tool_description}",
                "parameters": self.parameters,
            }
        }
        return tool_call_format
