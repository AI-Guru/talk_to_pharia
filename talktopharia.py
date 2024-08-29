import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import gradio as gr

# Define model and tokenizer.
MODEL_ID = "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)

# Move model to GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Defince the default system message.
default_system_message = "Du bist ein hilfreicher Assistent. Du gibst ansprechende, gut strukturierte Antworten auf Benutzeranfragen."

# Function to generate response with chat history
def generate_response(system_message, chat_history, user_input):
    # If system_message is present, prepend it to the chat history at the beginning.
    if chat_history == "" or chat_history is None:
        chat_history = f"<|start_header_id|>system<|end_header_id|>\n{system_message}<|eot_id|>"

    # Append the user input to the chat history.
    chat_history += f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    # Generate response.
    inputs = tokenizer(chat_history, return_token_type_ids=False, return_tensors="pt").to(device)
    eos_token_id = tokenizer("<|endoftext|>", return_tensors="pt").input_ids.to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024, eos_token_id=eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)

    # Do some cleanup. Not sure why this happens.
    chat_history = generated_text
    chat_history = chat_history.replace("<|start_header_id|> system<|end_header_id|>", "<|start_header_id|>system<|end_header_id|>").strip()
    chat_history = chat_history.replace("<|start_header_id|> user<|end_header_id|>", "<|start_header_id|>user<|end_header_id|>").strip()
    chat_history = chat_history.replace("<|start_header_id|> assistant<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>").strip()

    response_markdown = chat_history.strip()
    response_markdown = response_markdown.replace("<|start_header_id|>system<|end_header_id|>", "**System:**")
    response_markdown = response_markdown.replace("<|start_header_id|>user<|end_header_id|>", "\n**User:**")
    response_markdown = response_markdown.replace("<|start_header_id|>assistant<|end_header_id|>", "\n**Assistant:**")
    response_markdown = response_markdown.replace("<|eot_id|>", "\n")
    response_markdown = response_markdown.replace("<|endoftext|>", "\n")

    return response_markdown, chat_history

# Gradio interface function
def chatbot(system_message, user_input, chat_history=""):
    response, updated_history = generate_response(system_message, chat_history, user_input)
    return response, updated_history, ""

# Gradio Interface with chat history and system message
with gr.Blocks() as interface:
    with gr.Column():
        gr.Markdown("# Rede mit Pharia")
        system_message = gr.Textbox(default_system_message, label="System Message", placeholder="Enter system message here...")
        chat_history = gr.State()
        assert chat_history is not None
        
        with gr.Row():
            user_input = gr.Textbox(label="Benutzereingabe", placeholder="Red mit mir.")
            output = gr.Markdown(label="Antwort")
        
        user_input.submit(chatbot, [system_message, user_input, chat_history], [output, chat_history, user_input])

    # Launch the Gradio app
    interface.launch()

