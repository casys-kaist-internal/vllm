import statistics

base_file_path = "/home/sjchoi/workspace/vllm/benchmarks/base_{}.txt"
sps_file_path = "/home/sjchoi/workspace/vllm/benchmarks/sps_{}_{}.txt"

for batch_size in range(1, 25):
    base_file = open(base_file_path.format(batch_size), "r")
    lines = base_file.readlines()[3:]  # Exclude the first 3 lines
    base_file.close()

    latencies = []
    for line in lines:
        elements = line.strip().split(",")
        if len(elements) >= 3 and elements[1].strip() == str(batch_size):
            latencies.append(float(elements[2]))
        else:
            original_avg_latency = float(line.strip().split(" ")[-2])

    # Remove outliers from latencies
    latencies = [x for x in latencies if abs(
        x - statistics.mean(latencies)) <= 2 * statistics.stdev(latencies)]

    original_target = sum(latencies) / len(latencies)

    for draft_size in range(2, 5):
        sps_file = open(sps_file_path.format(batch_size, draft_size), "r")
        lines = sps_file.readlines()[3:]
        sps_file.close()

        draft_latencies = []
        target_latencies = []

        for line in lines:
            elements = line.strip().split(",")
            if len(elements) >= 3:
                if elements[0] == "DRAFT_DECODE":
                    if elements[1].strip() == str(batch_size):
                        draft_latencies.append(float(elements[2]))
                elif elements[0] == "TARGET_DECODE":
                    if elements[1].strip() == str(batch_size * (1 + draft_size)):
                        target_latencies.append(float(elements[2]))
            else:
                avg_latency = float(line.strip().split(" ")[-2])

        # Remove outliers from target_latencies and draft_latencies
        target_latencies = [x for x in target_latencies if abs(
            x - statistics.mean(target_latencies)) <= 2 * statistics.stdev(target_latencies)]
        draft_latencies = [x for x in draft_latencies if abs(
            x - statistics.mean(draft_latencies)) <= 2 * statistics.stdev(draft_latencies)]

        draft_latency = sum(draft_latencies) / len(draft_latencies)
        target_latency = sum(target_latencies) / len(target_latencies)

        print(
            f"{batch_size}, {draft_size}, {original_target:.5f}, {draft_latency:.5f}, {target_latency:.5f}, {original_avg_latency:.5f}, {avg_latency:.5f}")
