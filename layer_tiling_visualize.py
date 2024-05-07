

if __name__ == "__main__":
    
    # Load layer_tiling_results.json
    import json
    with open("layer_tiling_results_decoderlayer_0.json", "r") as f:
        time_results = json.load(f)
        
    
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
    
    # set size 
    fig.set_size_inches(20,15)
    
    color = 'tab:red'
    ax1.set_xlabel('input_tok_len')
    ax1.set_ylabel('time_taken', color=color)
    ax1.plot(input_tok_len, time_taken, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('throughput', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(input_tok_len, throughput, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # If throughput values are smaller than previous value, than plot it as red dot with value
    for i in range(1, len(throughput)):
        if throughput[i] < throughput[i-1]:
            ax2.plot(input_tok_len[i], throughput[i], 'ro')
            ax2.text(input_tok_len[i], throughput[i], f"{input_tok_len[i]:.2f}")
            
            
    # Force 0 to be in y axis
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # save
    plt.savefig("layer_tiling_visualize.png")