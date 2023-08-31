from vllm import LLM, SamplingParams, SpeculativeDecoder
from vllm.entrypoints.llm import LLM, SpeculativeDecoder
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

sps = SpeculativeDecoder(target_model="facebook/opt-125m", draft_model="facebook/opt-125m")
# Create an LLM.
#llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

outputs = sps.generate(prompts, sampling_params)
# Print the outputs.
#for output in outputs:
#    prompt = output.prompt
#    generated_text = output.outputs[0].text
#    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
