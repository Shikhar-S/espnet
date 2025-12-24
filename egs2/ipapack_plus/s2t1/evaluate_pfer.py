import re
import os
from ast import literal_eval
import sys
import json
import numpy as np
import panphon.distance
from collections import defaultdict
from tqdm import tqdm

dst = panphon.distance.Distance()


def compute_pfer(path):
    results = defaultdict(dict)
    with open(path,'r') as f:
        for line in f.readlines():  
            idx, pred = line.split("\t")
            if 'hyp=' in pred:
                category = 'hyp'
            else:
                category = 'ref'

            pred = pred.replace('hyp=','').replace('ref=','')
            pred = ''.join(list(literal_eval(pred)))
            results[idx][category] = pred

    pfers = []
    for k,v in tqdm(results.items()):
        pfer = dst.feature_edit_distance(v['hyp'].replace("g","ɡ"),v['ref'].replace("g","ɡ"))
        pfers.append(pfer)

    q75, q25 = np.percentile(pfers, [75 ,25])
    metrics = {"PFER mean": round(np.mean(pfers),3), 
               "PFER median": round(np.median(pfers),3), 
               "PFER IQR": round(q75 - q25,3),
               "PFER std": round(np.std(pfers),3)}
    return metrics


def parse_file(file_path):

    if 'frame' not in file_path:
        pattern = re.compile(r"recogs-(?P<dataset>[a-zA-Z0-9-]+)-iter-(?P<iterations>\d+)-avg-(?P<avg_checkpoints>\d+)-use-averaged-model\.txt")
    else:
        pattern = re.compile(r"recogs-(?P<dataset>[a-zA-Z0-9-]+)-greedy_search-iter-(?P<iterations>\d+)-avg-(?P<avg_checkpoints>\d+)")


    
    match = pattern.search(file_path)

    dataset = match.group("dataset")
    iterations = int(match.group("iterations"))
    avg_checkpoints = int(match.group("avg_checkpoints"))
    
    
    parsed_data = {
        "dataset": dataset,
        "iterations": iterations,
        "avg_checkpoints": avg_checkpoints
    }

    # Convert parsed data into a JSON object and return it
    return parsed_data


if __name__ == "__main__":

    base_path = sys.argv[1]
    base_folder = sys.argv[2]

    model_name = os.path.basename(base_path)

    output_path = os.path.join(base_path,base_folder)

    output_files = [i for i in os.listdir(output_path) if 'recogs' in i]

    with open(os.path.join(output_path,"final_metrics.json"),'w') as out:
        for output_file in output_files:
            meta_info = parse_file(output_file)
            meta_info['model'] = model_name
            path = os.path.join(output_path,output_file)
            metrics = compute_pfer(path)
            out.write(json.dumps({**metrics, **meta_info})+"\n")
            
        
        
    
    
