import torch
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from torch.profiler import profile, record_function, ProfilerActivity
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

batch_size = 10
steps = 50
#pipe.unet.set_attn_processor(AttnProcessor2_0())
# torch._inductor.config.trace.enabled = True
pipe.unet = torch.compile(pipe.unet)
prompt = "a photo of an astronaut riding a horse on mars"
print("after compile step", flush=True)
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("diffusion_run"):
        image = pipe(prompt, num_inference_steps=steps, num_images_per_prompt=batch_size).images[0]

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
#prof.export_chrome_trace("trace_rocm_latest.json")
