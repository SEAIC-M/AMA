

from datasets import load_dataset
import pandas as pd
import json
import re
import ast
import random
import os
PRIVACY_KEY_LIST = ["SURNAME","DRIVERLICENSENUM","PASSPORTNUM","TELEPHONENUM","CREDITCARDNUMBER","SOCIALNUM","EMAIL","GENDER","STREET","TAXNUM","CITY"]

def get_privacy_data():
    dataset = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy",split="validation")
    test_data = dataset.to_pandas()
    test_data = test_data[test_data['language'] == 'en']
    print(len(test_data))
    os.makedirs("./privacy_data", exist_ok=True)
    test_data.to_csv("./privacy_data/all_privacy_data.csv", index=False)

    # for loading the data from the csv file
    df = pd.read_csv("./privacy_data/all_privacy_data.csv")
    privacy_mask_values = df["privacy_mask"].tolist()
    privacy_data = {}

    for row in privacy_mask_values:
        row = re.sub(r'\}\s*\{', '}, {', row)
    
        row = ast.literal_eval(row)
        for value in row:
            label = value["label"]
            v = value["value"]

            if label not in privacy_data:
                privacy_data[label] = []
            privacy_data[label].append(v)

    for key, value in privacy_data.items():
        privacy_data[key] = list(set(value))
        privacy_data[key] = privacy_data[key]
   

    num_craft_privacy_info = 1000
    privacy_data_list = []


    for i in range(num_craft_privacy_info):
        privacy_data_dict = {}
        for key, value in privacy_data.items():
            if key not in PRIVACY_KEY_LIST:
                continue
            privacy_data_dict[key] = random.choice(value)
        privacy_data_list.append(privacy_data_dict)

    with open("./privacy_data/privacy_data.jsonl", "w", encoding="utf-8") as f:
        for item in privacy_data_list:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    get_privacy_data()