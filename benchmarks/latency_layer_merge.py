import csv

# 입력 CSV 파일 경로
input_csv_path = "/home/yhkim/workspace/vllm/benchmarks/result_per_layer/facebook/opt-125m_1.csv"
# 출력 CSV 파일 경로
output_csv_path = "/home/yhkim/workspace/vllm/benchmarks/result_per_layer_merged/facebook/opt-125m_1.csv"

# 결과를 저장할 딕셔너리
results = {}

# 입력 CSV 파일 읽기
with open(input_csv_path, mode='r') as file:
    reader = csv.reader(file)
    current_context_len = None
    for row in reader:
        if 'latency' in row[0] :
            if 'Avg latency' in row[0]:
                continue
            else:
                # 'latency' 줄 발견 시, 새 context_len로 초기화
                _, batch_size, context_len, latency = row
                current_context_len = context_len
                results[current_context_len] = {'batch_size': batch_size, 'PagedAttention': 0,'Activation': 0, 'LayerNorm': 0, 'Linear': 0, 'ResidualAdd': 0}
        elif current_context_len is not None:
            # 현재 context_len에 대한 레이어 시간 누적
            layer_type, layer_time = row[0], float(row[1])
            if 'Activation' in layer_type:
                results[current_context_len]['Activation'] += layer_time
            elif 'PagedAttention' in layer_type:
                results[current_context_len]['PagedAttention'] += layer_time
            elif 'LayerNorm' in layer_type:
                results[current_context_len]['LayerNorm'] += layer_time
            elif 'Linear' in layer_type:
                results[current_context_len]['Linear'] += layer_time
            elif 'ResidualAdd' in layer_type:
                results[current_context_len]['ResidualAdd'] += layer_time

# 출력 CSV 파일 쓰기
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 헤더 작성
    writer.writerow(['batch_size', 'context_len', 'PagedAttention', 'Activation', 'LayerNorm', 'Linear', 'ResidualAdd'])
    # 정리된 결과 작성
    for context_len, data in results.items():
        writer.writerow([data['batch_size'], context_len,data['PagedAttention'], data['Activation'], data['LayerNorm'], data['Linear'], data['ResidualAdd']])

