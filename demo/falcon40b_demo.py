import torch
import time
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

#model = AutoModelForCausalLM.from_pretrained("/data/opt66b", torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b-instruct", torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained("/data/opt66b", use_fast=False)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
#tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct", use_fast=True)

def run_prompt(base_prompt, input_prompt, max_length=1024, prompting=True):
    if prompting:
        input_prompt = "Prompt: " + input_prompt + "\nResponse:"
    #prompt = base_prompt+input_prompt
    prompt = input_prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    streamer = TextStreamer(tokenizer)
    #print("INFO: Sending prompt to opt-66b model")
    print("-----------------------------------------------------------")
    #_ = model.generate(input_ids, do_sample=True, streamer=streamer, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    generated_ids = model.generate(input_ids, do_sample=True, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    print("-----------------------------------------------------------")
    #return
    return (tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


def slow_print(response):
    #print("INFO: Printing response:")
    print("------------------------")
    print("Response")
    for line in str.split(response, '\n'):
        print(line, flush=True)
        time.sleep(.05)
    print("------------------------")


loop = 10
max_length = [256]
times = []
for max in max_length:
    for i in range(loop):
        os.system('clear')
        print("INFO: Loaded falcon-40b-instruct model onto MI300X")
        time.sleep(1)
        #for i in tqdm(range(0, 14), desc ="Loading checkpoint shards", bar_format = "{l_bar}|{bar}| {n_fmt}/{total_fmt}{postfix}"):
        #    time.sleep(.5)
        print("")
        print("Please enter your prompt: ", end='', flush=True)
        time.sleep(1)
        #input_prompt="Write me a sonnet about San Francisco"
        input_prompt="Write me a nice poem about San Francisco and AMD"
        input_prompt = input_prompt + "\n"
        for charac in input_prompt:
            print(charac, end='', flush=True)
            time.sleep(.05)
        print("")

        start = time.time()
        response = run_prompt(poem_prompt, input_prompt, max, prompting=False)
        print(response)
        end = time.time()
        time.sleep(5)
        times.append(end-start)

print("INFO: Demo has finished.")
print("INFO: Run times are {}".format(times))
