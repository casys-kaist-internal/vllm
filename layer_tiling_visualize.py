import os
import re

# default dictionary
from collections import defaultdict

if __name__ == "__main__":
    
    # Load layer_tiling_results.json
    import json
    
    profile_dir = "tiling_profile_results_json/"
    
    gpu_names = set()
    model_names = set()
    
    gpus_per_model = defaultdict(list)
    for json_name in os.listdir(profile_dir):
        if not json_name.endswith(".json"):
            continue
        gpu_name = re.search(r"tiling_test_gpu_(.*)_model_", json_name).group(1)
        gpu_names.add(gpu_name)

        model_name = re.search(r"model_(.*).json", json_name).group(1)
        model_names.add(model_name)
        gpus_per_model[model_name].append(gpu_name)


    for model_name in model_names:
        time_results_map = {}
        
        print(f"Model: {model_name}")
        
        gpu_names = gpus_per_model[model_name]
        
        for json_name in os.listdir(profile_dir):
            if not json_name.endswith(".json"):
                continue
            gpu_name = re.search(r"tiling_test_gpu_(.*)_model_", json_name).group(1)
            s_model_name = re.search(r"model_(.*).json", json_name).group(1)
            
            file_name = profile_dir + json_name
            
            if model_name != s_model_name:
                continue
            
            with open(file_name, "r") as f:
                time_results = json.load(f)    
            time_results_map[gpu_name] = time_results

        # time_results is list of {input_tok_len, time_taken, max_query_len, throughput}        
        # x axis : input_tok_len
        # y axis 1 : time_taken
        # y axis 2 : throughput
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # sort results by input_tok_len
        time_results = sorted(time_results, key=lambda x: x["input_tok_len"])
        
        input_tok_len = [result["input_tok_len"] for result in time_results]
        time_taken = [result["time_taken"] for result in time_results]
        throughput = [result["throughput"] for result in time_results]
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        # cool tone
        throughput_colours = ['b', 'c', 'm']
        # warm tone
        latency_colours = ['r', 'y', 'brown']

        # set size 
        fig.set_size_inches(20,15)
        
        # Set title
        ax1.set_title(f"Tiling Test: {model_name}")
        
        ax1.set_xlabel('input_tok_len')
        ax1.set_ylabel('time_taken')
        ax2.set_ylabel('throughput')

        for i, gpu_name in enumerate(gpu_names):
            time_results = time_results_map[gpu_name]
            time_results = sorted(time_results, key=lambda x: x["input_tok_len"])
            
            input_tok_len = [result["input_tok_len"] for result in time_results]
            time_taken = [result["time_taken"] for result in time_results]
            throughput = [result["throughput"] for result in time_results]
            
            ax1.plot(input_tok_len, time_taken, color=latency_colours[i])
            ax1.tick_params(axis='y', labelcolor=latency_colours[i])
            
            # thickness 2
            ax2.plot(input_tok_len, throughput, color=throughput_colours[i], linewidth=4)
            ax2.tick_params(axis='y', labelcolor=throughput_colours[i])
            
        
        # If throughput values are smaller than previous value, than plot it as red dot with value
        # for i in range(1, len(throughput)):
        #     if throughput[i] < throughput[i-1]:
        #         ax2.plot(input_tok_len[i], throughput[i], 'ro')
        #         ax2.text(input_tok_len[i], throughput[i], f"{input_tok_len[i]:.2f}")
        
        # labels
        ax1.legend(gpu_names, loc='upper left')
        ax2.legend(gpu_names, loc='upper right')
                
                
        # Force 0 to be in y axis
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # save
        print(f"Saving {profile_dir}tiling_test_{model_name}.png")
        plt.savefig(profile_dir + f"tiling_test_{model_name}.png")
        plt.clf()