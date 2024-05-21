from random import randint
from typing import Any, Callable, Generic, Iterator, Tuple, TypeVar, Union
import typing

from einops import rearrange
import torch
from transformers.models.clip import CLIPTokenizer, CLIPTextModel

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .inputs import LabeledVid, TrainingVideo

class InputScheduler(Iterator[int]):
    steps: int
    items: int
    i: int

    def __init__(self, steps: int, items: int):
        self.steps = steps
        self.items = items
        self.i = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        raise StopIteration

def tensor_to_vae_latent(t: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w") 
    latents = vae.encode(t).latent_dist.sample() # type: ignore
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

S, T = TypeVar("S", bound=InputScheduler), TypeVar("T")
class InputSchedule(Generic[S, T]):
    scheduler: type[S]
    schedule: Union[S, None]
    items: list[T]

    def __init__(self, scheduler: type[S], items: list[T]):
        self.scheduler = scheduler
        self.schedule = None
        self.items = items

    def train(self, steps: int):
        self.schedule = self.scheduler(steps, len(self.items))
        return self

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.schedule is not None:
            i = next(self.schedule)
            print(f"DEBUG: Choosing video #{i}")
            return self.items[i]
        
        raise StopIteration
    
    @staticmethod
    def prepare_dataset(
        sched: "InputSchedule[S, LabeledVid]",
        vae: AutoencoderKL,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        device: Any,
    ) -> "InputSchedule[S, TrainingVideo]":
        """
        Prepares all of the training data for model training,
        returning a new InputSchedule with the processed data.

        Run me in a torch.no_grad() environment!
        """
        # Prompt Encoding.
        text_prompts = map(lambda vid: vid.prompt, sched.items)
        prompt_ids = [tokenizer(
            [txt], 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(device) for txt in text_prompts]

        #text encoding
        text_encoder.to(device)
        encoder_hidden_states_list: list[torch.Tensor] = [text_encoder(prompt_id)[0] for prompt_id in prompt_ids]
        print("ENCODER STATES LIST: ", encoder_hidden_states_list)
        text_encoder.to('cpu') # type: ignore

        def to_latent(px: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            print("input batch shape:", px.shape)
            px = px * 2.0 - 1.0 #normalize to the expected range (-1, 1)
            px = px.permute(0, 3, 1, 2).unsqueeze(0)#B,H,W,C to B,F,C,H,W
            px = px.to(device)
            latents = tensor_to_vae_latent(px, vae)
            return (px, latents)

        pixel_values = list(map(lambda vid: vid.frames, sched.items))
        print(f"Received {len(pixel_values)} batche(s):")
        pixel_list, latent_list = [list(s) for s in zip(*map(to_latent, pixel_values))]
        pixel_list = typing.cast(list[torch.Tensor], pixel_list)
        latent_list = typing.cast(list[torch.Tensor], latent_list)
        
        batch_size = len(pixel_list)
        
        print("batch_size:", batch_size)
        vae.to('cpu')

        processed_inputs = list(map(lambda t: TrainingVideo.new(*t), zip(encoder_hidden_states_list, pixel_list, latent_list)))

        return InputSchedule(sched.scheduler, processed_inputs)

class RoundRobinSchedule(InputScheduler):
    def __next__(self):
        if self.i >= self.steps:
            raise StopIteration
        
        i = self.i % self.items
        self.i += 1

        return i
    
class InOrderProportionalSchedule(InputScheduler):
    div: int

    def __init__(self, steps: int, items: int):
        super().__init__(steps, items)
        self.div = max(steps // items, 1)

    def __next__(self):
        if self.i >= self.steps:
            raise StopIteration
        
        i_item = (self.i // self.div) % self.items
        self.i += 1

        return i_item 
    
class RandomSchedule(InputScheduler):
    def __next__(self):
        if self.i >= self.steps:
            raise StopIteration

        i = randint(0, self.items -1)
        self.i += 1

        return i
        
INPUT_SCHEDULERS: dict[str, type[InputScheduler]] = {
    "round-robin": RoundRobinSchedule,
    "in-order-proportional": InOrderProportionalSchedule,
    "random": RandomSchedule,
}