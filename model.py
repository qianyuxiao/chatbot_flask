from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
from threading import Thread
import torch


# Set model and tokenizer
def get_model_and_tokenizer(
    llm_model_name = "HuggingFaceH4/zephyr-7b-beta",
    torch_dtype = torch.bfloat16):
    
    # Set the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    torch.cuda.set_device(device)

    # initiate model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(llm_model_name,
                                                torch_dtype=torch_dtype, 
                                                load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name,
                                            torch_dtype=torch_dtype)
    return model, tokenizer

# define stream function
def z_stream(user_prompt,model,tokenizer):

    # system_prompt = 'You are ECG AI, an intelligent assistant dedicated to providing effective solutions. Your responses will include engaging touch. Analyze user queries and provide clear and practical answers. Focus on delivering solutions that are accurate, actionable, and helpful. If additional information is required for a more precise solution, politely ask clarifying questions. Your goal is to assist users by providing effective and reliable solutions to their queries.'
    system_prompt = 'You are ECG AI, an intelligent assistant dedicated to providing effective solutions. Your responses will include engaging touch. Analyze user queries and provide clear and practical answers. Focus on delivering solutions that are accurate, actionable, and helpful. If you have no solution or additional information is required for a more precise solution, politely ask clarifying questions but make it short. Your goal is to assist users by providing effective and reliable solutions to their queries.'

    E_INST = "</s>"
    user, assistant = "<|user|>", "<|assistant|>"

    prompt = f"{system_prompt}{E_INST}\n{user}\n{user_prompt.strip()}{E_INST}\n{assistant}\n"

    inputs = tokenizer([prompt], return_tensors="pt").to('cuda')

    streamer = TextIteratorStreamer(tokenizer,  skip_prompt=True, skip_special_tokens=True)

    # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1000)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    # generated_text = ""
    # for _, new_text in enumerate(streamer):
    #     generated_text += new_text
    # return generated_text
    for _, new_text in enumerate(streamer):
        yield new_text