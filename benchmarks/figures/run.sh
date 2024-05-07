batch_size=16

for dataset in "gsm8k" "humaneval" "alpaca" "mt-bench" "sharegpt"; do
    # Loop through draft size from 4 to 8
    rm "result/fig1_sps_$dataset.csv"> "/dev/null"
    for draft_size in {4..8}; do
        echo "batch_size=$batch_size, draft_size=$draft_size" >> "result/fig1_sps_$dataset.csv"
        python3 figure1.py \
        --dataset "$dataset" \
        --engine sps \
        --num-iters 10 \
        --draft-size "$draft_size" \
        --batch-size "$batch_size" \
        >> "result/fig1_sps_$dataset.csv"
    done
    echo "Done $dataset $batch_size"
done