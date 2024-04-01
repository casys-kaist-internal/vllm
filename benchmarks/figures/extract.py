def extract_output_from_text(text):
    lines = text.split('\n')
    output_texts = []
    
    reading_output = False
    current_output = ""
    
    # Initialize a cycle of numbers from 2 to 8
    from itertools import cycle
    numbers_cycle = cycle(range(2, 9))
    current_number = next(numbers_cycle)  # Start the cycle
    
    for line in lines:
        if 'Output:' in line:
            reading_output = True
            # Include the current number with the output
            current_output = f"{current_number} {line.split('Output:', 1)[1].strip()}"
            current_number = next(numbers_cycle)  # Move to the next number
        elif reading_output:
            current_output += " " + line.strip()
        if reading_output and ('Input:' in line or line == lines[-1]):
            reading_output = False
            if current_output:
                output_texts.append(current_output.strip())
                current_output = ""
    
    return output_texts

def read_file_and_extract_outputs(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    extracted_outputs = extract_output_from_text(text)
    return extracted_outputs

# Replace 'input.txt' with the path to your actual text file
file_path = './result/s_out_sps_sharegpt.csv'
extracted_outputs = read_file_and_extract_outputs(file_path)

for output in extracted_outputs:
    print(output)
