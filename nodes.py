import os
import random
import logging
import datetime
import itertools

from dataclasses import dataclass
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler # type: ignore
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer
from dataclasses import dataclass

from .animatediff.models.unet import UNet3DConditionModel
from .animatediff.pipelines.pipeline_animation import AnimationPipeline
from .animatediff.utils.util import save_videos_grid, load_weights
from .animatediff.utils.lora_handler import LoraHandler
from .animatediff.utils.lora import extract_lora_child_module

from . import utils

from .motion_lora import MotionLoraInfo, MotionLoraList

from lion_pytorch import Lion
import comfy.model_management
import comfy.utils
import folder_paths

try:
    import xformers # type: ignore
    import xformers.ops # type: ignore

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False

import typing
from typing import Any, Generic, Iterable, Iterator, TypeVar, Union
import torch.amp

script_directory = os.path.dirname(os.path.abspath(__file__))
folder_paths.add_model_folder_path("animatediff_models", str(Path(__file__).parent.parent / "models"))
folder_paths.add_model_folder_path("animatediff_models", str(Path(folder_paths.models_dir) / "animatediff_models"))

def create_save_paths(output_dir: str):
    directories = [
        output_dir,
        f"{output_dir}/samples",
        f"{output_dir}/sanity_check",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def resize_and_pad_images(images, output_size):
    images = images.permute(0, 3, 1, 2)
    
    # Calculate aspect ratio of the output size
    aspect_ratio = output_size[1] / output_size[0]
    
    # Calculate resizing size that maintains aspect ratio
    batch_size, channels, height, width = images.size()
    if width / height > aspect_ratio:
        # If image is wider than desired aspect ratio, fit to width
        resize_width = output_size[1]
        resize_height = round(resize_width / width * height)
    else:
        # If image is taller than desired aspect ratio, fit to height
        resize_height = output_size[0]
        resize_width = round(resize_height / height * width)
    
    # Resize the images while maintaining the aspect ratio
    resized_images = F.interpolate(images, size=(resize_height, resize_width), mode='bilinear', align_corners=False)

    # Calculate the padding required to make the images the same size
    pad_width = max(0, output_size[1] - resized_images.size(3))
    pad_height = max(0, output_size[0] - resized_images.size(2))
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    # Pad the images to the desired size with black bars
    padded_images = F.pad(resized_images, (pad_left, pad_right, pad_top, pad_bottom), value=0)

    return padded_images

def do_sanity_check(
    sanity_check: list[torch.Tensor],  
    output_dir: str,
    text_prompt: list[str],
):
    resized_images = list(map(lambda img: resize_and_pad_images(img, (512, 512)), sanity_check))
    sanity_check = torch.cat(resized_images, dim=0) # type: ignore
    sanity_check = sanity_check.unsqueeze(0) # type: ignore
        
    sanity_check = sanity_check * 2.0 - 1.0 #normalize to the expected range (-1, 1) # type: ignore
    

    sanity_check, texts = sanity_check.cpu(), text_prompt  # type: ignore
    sanity_check = rearrange(sanity_check, "b f c h w -> b c f h w")
    for idx, (pixel_value, text) in enumerate(zip(sanity_check, texts)):
        pixel_value = pixel_value[None, ...]
        text = text
        save_name = f"{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'-{idx}'}.mp4"
        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{save_name}", rescale=False)
    return typing.cast(torch.Tensor, sanity_check)

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None # type: ignore
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }

def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name,
        "params": params,
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params

def create_optimizer_params(model_list, lr):
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(
                params=itertools.chain(*model),
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        if is_lora and condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters(): # type: ignore
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue
            
                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params

def scale_loras(lora_list: list, scale: float, step=None): 
    # Assumed enumerator
    if step is not None:
        process_list = range(0, len(lora_list), 1)
    else:
        process_list = lora_list

    for lora_i in process_list:
        if step is not None:
            lora_list[lora_i].scale = scale
        else:
            lora_i.scale = scale # type: ignore


def get_spatial_latents(
        pixel_values: torch.Tensor, 
        noisy_latents:torch.Tensor, 
        target: torch.Tensor,
    ):
    ran_idx = torch.randint(0, pixel_values.shape[2], (1,)).item()

    noisy_latents_input = None
    target_spatial = None
    
    noisy_latents_input = noisy_latents[:, :, ran_idx, :, :] # type: ignore
    target_spatial = target[:, :, ran_idx, :, :] # type: ignore

    return noisy_latents_input, target_spatial

def create_ad_temporal_loss(
        model_pred: torch.Tensor, 
        loss_temporal: torch.Tensor, 
        target: torch.Tensor
    ):

    beta = 1
    alpha = (beta ** 2 + 1) ** 0.5

    ran_idx = torch.randint(0, model_pred.shape[2], (1,)).item()

    model_pred_decent = alpha * model_pred - beta * model_pred[:, :, ran_idx, :, :].unsqueeze(2) # type: ignore
    target_decent = alpha * target - beta * target[:, :, ran_idx, :, :].unsqueeze(2) # type: ignore

    loss_ad_temporal = F.mse_loss(model_pred_decent.float(), target_decent.float(), reduction="mean")
    loss_temporal = loss_temporal + loss_ad_temporal

    return loss_temporal



class ADMD_Ext_InputPipeStart:
    """
    Start of multi-video training.
    """

    RETURN_TYPES = ("ADMDDATAPIPELINE", )
    RETURN_NAMES = ("pipe_out", )
    FUNCTION = "process"

    CATEGORY = "AD_MotionDirector"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_scheduler": (list(utils.scheduling.INPUT_SCHEDULERS.keys()), {"default": "round-robin"})
        }}
    
    def process(self, input_scheduler: str):
        return (utils.VideoPipe.empty(utils.scheduling.INPUT_SCHEDULERS[input_scheduler]),)

class ADMD_Ext_InputPipeline:
    """
    Allows for training of multiple videos.
    """

    RETURN_TYPES = ("ADMDDATAPIPELINE",)
    RETURN_NAMES = ("pipe_out", )
    FUNCTION = "process"

    CATEGORY = "AD_MotionDirector"
    # OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ADMDDATAPIPELINE", ),
                "frames": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "",}),
            },

        }
    
    def process(self, pipe: utils.VideoPipe, frames, prompt: str) -> tuple[utils.VideoPipe, ]:
        print(f"Videos in pipe: {len(pipe) + 1}")
        vid  = utils.inputs.LabeledVid(frames, prompt)

        return (pipe.append(vid), )        

Sched = Union[DDIMScheduler, DDPMScheduler]

@dataclass
class _Pipeline:
    validation_pipeline: AnimationPipeline
    train_noise_scheduler: Sched
    train_noise_scheduler_spatial: Sched
    unet: UNet3DConditionModel
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer

@dataclass
class _ADMD_Pipeline: 
    optimizer_temporal: Union[torch.optim.AdamW, Lion]
    lr_scheduler_temporal: torch.optim.lr_scheduler.LambdaLR
    optimizer_spatial_list: list[Union[torch.optim.AdamW, Lion]]
    lr_scheduler_spatial_list: list[torch.optim.lr_scheduler.LambdaLR]
    unet: UNet3DConditionModel
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    tokenizer: CLIPTokenizer

    inputs: utils.scheduling.InputSchedule[Any, utils.inputs.LabeledVid]

    train_noise_scheduler: Sched
    train_noise_scheduler_spatial: Sched
    validation_pipeline: AnimationPipeline
    global_step: int
    max_train_steps: int
    scaler: torch.amp.GradScaler # type: ignore
    include_resnet: bool
    seed: int

class ADMD_InitializeTraining:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipeline": ("PIPELINE", ),
            "lora_name": ("STRING", {"multiline": False, "default": "motiondirectorlora",}),
            "input_pipeline": ("ADMDDATAPIPELINE", ),
            "max_train_steps": ("INT", {"default": 300, "min": 0, "max": 100000, "step": 1}),
            "learning_rate": ("FLOAT", {"default": 5e-4, "min": 0, "max": 10000, "step": 0.00001}),
            "learning_rate_spatial": ("FLOAT", {"default": 1e-4, "min": 0, "max": 10000, "step": 0.00001}),    
            "lora_rank": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 8}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            
            "optimization_method": (
            [   
                'Lion',
                'AdamW',
            ], {
               "default": 'Lion'
            }),
            "include_resnet": ("BOOLEAN", {"default": False}),
            },
            
            }
    
    RETURN_TYPES = ("IMAGE", "ADMDPIPELINE", "LORAINFO")
    RETURN_NAMES =("sanitycheck", "admd_pipeline", "lora_info",)
    FUNCTION = "process"

    CATEGORY = "AD_MotionDirector"

    def process(self, pipeline: _Pipeline, input_pipeline: utils.VideoPipe,  
                lora_name, learning_rate, learning_rate_spatial, 
                lora_rank, seed, optimization_method, max_train_steps, include_resnet):
        with torch.inference_mode(False):
                      
            validation_pipeline = pipeline.validation_pipeline
            train_noise_scheduler = pipeline.train_noise_scheduler
            train_noise_scheduler_spatial = pipeline.train_noise_scheduler_spatial

            unet = pipeline.unet
            text_encoder = pipeline.text_encoder
            vae = pipeline.vae
            tokenizer = pipeline.tokenizer

            torch.manual_seed(seed)


            print("INPUT SCHEDULER: ", input_pipeline.input_scheduler)
            input_schedule = utils.scheduling.InputSchedule(input_pipeline.input_scheduler, input_pipeline.videos)


            scale_lr = False
            lr_warmup_steps = 0
            lr_scheduler = "constant"

            train_batch_size = 1
            adam_beta1 = 0.9
            adam_beta2 = 0.999
            adam_weight_decay = 1e-2
            gradient_accumulation_steps = 1
            
            is_debug = False

            lora_unet_dropout = 0.1
            if include_resnet:
                target_spatial_modules = ["Transformer3DModel", "ResnetBlock2D"]
            else: 
                target_spatial_modules = ["Transformer3DModel"]

            target_temporal_modules = ["TemporalTransformerBlock"]          
  
            name = lora_name
            date_calendar = datetime.datetime.now().strftime("%Y-%m-%d")
            date_time = datetime.datetime.now().strftime("%H-%M-%S")
            folder_name = "debug" if is_debug else name + date_time
            
            output_dir = os.path.join(script_directory, "outputs", date_calendar, folder_name)

            if is_debug and os.path.exists(output_dir):
                os.system(f"rm -rf {output_dir}")

            # Make one log on every process with the configuration for debugging.
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
            )

            # set paths
            spatial_lora_path = os.path.join(folder_paths.models_dir,"loras", "trained_spatial", date_calendar, date_time, lora_name) 
            temporal_lora_path = os.path.join(folder_paths.models_dir,"animatediff_motion_lora", date_calendar, date_time, lora_name)
            temporal_lora_base_path = os.path.join(date_calendar, date_time, lora_name)
            
            lora_info = {
                "lora_name": lora_name,
                "lora_rank": lora_rank,
                "spatial_lora_path": spatial_lora_path,
                "temporal_lora_path": temporal_lora_path,
                "temporal_lora_base_path": temporal_lora_base_path
            }

            if optimization_method == "AdamW":   
                print("Using AdamW optimizer for training") 
                optimizer = torch.optim.AdamW
            else:
                print("Using Lion optimizer for training")
                optimizer = Lion
                learning_rate, learning_rate_spatial = map(lambda lr: lr / 10, (learning_rate, learning_rate_spatial))
                adam_weight_decay *= 10

            if scale_lr:
                learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size)

            # Temporal LoRA
            
            # one temporal lora
            lora_manager_temporal = LoraHandler(use_unet_lora=True, unet_replace_modules=target_temporal_modules)
            
            unet_lora_params_temporal, unet_negation_temporal = lora_manager_temporal.add_lora_to_model(
                True, unet, lora_manager_temporal.unet_replace_modules, 0,
                temporal_lora_path, r=lora_rank)

            optimizer_temporal = optimizer(
                create_optimizer_params([param_optim(unet_lora_params_temporal, True, is_lora=True,
                                                    extra_params={**{"lr": learning_rate}}
                                                    )], learning_rate),
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay
            )
        
            lr_scheduler_temporal = get_scheduler(
                lr_scheduler,
                optimizer=optimizer_temporal,
                num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
                num_training_steps=max_train_steps * gradient_accumulation_steps,
            )

            # Spatial LoRAs

            unet_lora_params_spatial_list = []
            optimizer_spatial_list = []
            lr_scheduler_spatial_list = []

            lora_manager_spatial = LoraHandler(use_unet_lora=True, unet_replace_modules=target_spatial_modules)
            unet_lora_params_spatial, unet_negation_spatial = lora_manager_spatial.add_lora_to_model(
                True, unet, lora_manager_spatial.unet_replace_modules, lora_unet_dropout,
                spatial_lora_path, r=lora_rank)

            unet_lora_params_spatial_list.append(unet_lora_params_spatial)

            optimizer_spatial = optimizer(
                create_optimizer_params([param_optim(unet_lora_params_spatial, True, is_lora=True,
                                                    extra_params={**{"lr": learning_rate_spatial}}
                                                    )], learning_rate_spatial),
                lr=learning_rate_spatial,
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay
            )

            optimizer_spatial_list.append(optimizer_spatial)

            # Scheduler
            lr_scheduler_spatial = get_scheduler(
                lr_scheduler,
                optimizer=optimizer_spatial,
                num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
                num_training_steps=max_train_steps * gradient_accumulation_steps,
            )
            lr_scheduler_spatial_list.append(lr_scheduler_spatial)
           
            # Support mixed-precision training
            if 'scaler' not in globals():
                scaler = torch.cuda.amp.GradScaler()
                print("initialize scaler")
            else:
                scaler.reset() # type: ignore
                print("reset scaler")

        admd_pipeline = _ADMD_Pipeline(
            optimizer_temporal=optimizer_temporal,
            optimizer_spatial_list=optimizer_spatial_list,
            lr_scheduler_spatial_list=lr_scheduler_spatial_list,
            lr_scheduler_temporal=lr_scheduler_temporal,
            unet=unet,
            text_encoder=text_encoder,
            vae=vae,
            inputs=input_schedule,
            tokenizer=tokenizer,
            train_noise_scheduler=train_noise_scheduler,
            train_noise_scheduler_spatial=train_noise_scheduler_spatial,
            validation_pipeline=validation_pipeline,
            global_step=0,
            max_train_steps=max_train_steps,
            scaler=scaler,
            include_resnet=include_resnet,
            seed=seed
        )

        #Data batch sanity check
        sanitycheck = do_sanity_check(
                sanity_check=list(map(lambda vid: vid.frames, input_schedule.items)),  
                output_dir=output_dir, 
                text_prompt=list(map(lambda vid: vid.prompt, input_schedule.items))
        )
 
        sanitycheck = sanitycheck.view(*sanitycheck.shape[1:])
        sanitycheck = sanitycheck.permute(1, 2, 3, 0).cpu()
        sanitycheck = (sanitycheck + 1.0) / 2.0
        return (sanitycheck, admd_pipeline, lora_info,)

class ADMD_DiffusersLoader:
    @classmethod
    def IS_CHANGED(cls):
        return ""
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required":
                {
                "additional_models": ("ADDITIONAL_MODELS", ),  
                "download_default": ("BOOLEAN", {"default": False},),
                "scheduler": (
            [   
                'DDIMScheduler',
                'DDPMScheduler',
            ], {
               "default": 'DDIMScheduler'
            }),
            "use_xformers": ("BOOLEAN", {"default": False}),
                },
                "optional": {
                 "model": (paths,),
                }
                
            }
    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "load_checkpoint"

    CATEGORY = "AD_MotionDirector"

    def load_checkpoint(self, download_default, scheduler, use_xformers, additional_models, model=""):
        with torch.inference_mode(False):
            device = comfy.model_management.get_torch_device()
            target_path = os.path.join(folder_paths.models_dir,'diffusers', "stable-diffusion-v1-5")      
            if download_default and model != os.path.exists(target_path):
                from huggingface_hub import snapshot_download
                download_to = os.path.join(folder_paths.models_dir,'diffusers')
                snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", ignore_patterns=["*.safetensors","*.ckpt", "*.pt", "*.png", "*non_ema*", "*safety_checker*", "*fp16*"], 
                                    local_dir=f"{download_to}/stable-diffusion-v1-5", local_dir_use_symlinks=False)   
                model_path = "stable-diffusion-v1-5"
            else:
                model_path = model
            
            for search_path in folder_paths.get_folder_paths("diffusers"):
                if os.path.exists(search_path):
                    path = os.path.join(search_path, model_path)
                    if os.path.exists(path):
                        model_path = path
                        break      
            
            config = OmegaConf.load(os.path.join(script_directory, f"configs/training/motion_director/training.yaml"))
       
            vae          = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
            tokenizer    = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
         
            unet_additional_kwargs = config.unet_additional_kwargs
            unet = UNet3DConditionModel.from_pretrained_2d(
                model_path, subfolder="unet", 
                unet_additional_kwargs=unet_additional_kwargs
            )
            
            # Load scheduler, tokenizer and models.
            noise_scheduler_kwargs = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "linear",
                'clip_sample':   False,
                'steps_offset': 1
            }
            # Determine the scheduler class based on the scheduler variable
            SchedulerClass = DDPMScheduler if scheduler == "DDPMScheduler" else DDIMScheduler
            print(f"using {SchedulerClass.__name__} for training")

            # Set the beta_schedule and create the default noise scheduler
            noise_scheduler_kwargs['beta_schedule'] = 'linear'
            noise_scheduler = SchedulerClass(**noise_scheduler_kwargs)

            # Set the beta_schedule for the spatial noise scheduler and create it
            noise_scheduler_kwargs['beta_schedule'] = 'scaled_linear'
            train_noise_scheduler_spatial = SchedulerClass(**noise_scheduler_kwargs)

            # Reset the beta_schedule for the linear noise scheduler and create it
            noise_scheduler_kwargs['beta_schedule'] = 'linear'
            train_noise_scheduler = SchedulerClass(**noise_scheduler_kwargs)
            
            # Freeze all models for LoRA training
            unet.requires_grad_(False)
            vae.requires_grad_(False) # type: ignore
            text_encoder.requires_grad_(False) # type: ignore

            #xformers
            if XFORMERS_IS_AVAILABLE:
                if use_xformers:
                    unet.enable_xformers_memory_efficient_attention()
                else:
                    unet.disable_xformers_memory_efficient_attention()

            # Enable gradient checkpointing
            unet.enable_gradient_checkpointing()

            # Validation pipeline
            validation_pipeline = AnimationPipeline(
                unet=unet, 
                vae=vae,                    # type: ignore
                tokenizer=tokenizer, 
                text_encoder=text_encoder,  # type: ignore
                scheduler=noise_scheduler,  # type: ignore
            ).to(device)

            motion_model, domain_adapter_path = additional_models 

            validation_pipeline = load_weights(
                validation_pipeline, 
                motion_model=motion_model,
                adapter_lora_path=domain_adapter_path, 
                dreambooth_model_path=""
            )

            validation_pipeline.enable_vae_slicing()         


            pipeline = _Pipeline(
                validation_pipeline=validation_pipeline,
                train_noise_scheduler=train_noise_scheduler,
                train_noise_scheduler_spatial=train_noise_scheduler_spatial,
                unet=unet,
                vae=vae,                            # type: ignore
                text_encoder=text_encoder,          # type: ignore
                tokenizer=tokenizer
            )

            return (pipeline,)

class ADMD_CheckpointLoader:
    @classmethod
    def IS_CHANGED(cls):
        return ""
    @classmethod
    def INPUT_TYPES(cls):

        return {"required":
                {
                "additional_models": ("ADDITIONAL_MODELS", ),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),  
                "scheduler": (
            [   
                'DDIMScheduler',
                'DDPMScheduler',
            ], {
               "default": 'DDIMScheduler'
            }),
            "use_xformers": ("BOOLEAN", {"default": False}),
                },                
            }
    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "load_checkpoint"

    CATEGORY = "AD_MotionDirector"

    def load_checkpoint(self, scheduler, use_xformers, additional_models, ckpt_name):
        with torch.inference_mode(False):            
            model_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
            ad_unet_config = OmegaConf.load(os.path.join(script_directory, f"configs/ad_unet_config.yaml"))

            from diffusers.loaders.single_file_utils import (convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint, create_text_encoder_from_ldm_clip_checkpoint, create_vae_diffusers_config, create_unet_diffusers_config)
            from safetensors import safe_open

            if model_path.endswith(".safetensors"):                                 # type: ignore
                dreambooth_state_dict = {}
                with safe_open(model_path, framework="pt", device="cpu") as f:      # type: ignore
                    for key in f.keys():
                        dreambooth_state_dict[key] = f.get_tensor(key)
            elif model_path.endswith(".ckpt"):                                      # type: ignore
                dreambooth_state_dict = torch.load(model_path, map_location="cpu")  # type: ignore
                while "state_dict" in dreambooth_state_dict:
                    dreambooth_state_dict = dreambooth_state_dict["state_dict"]

            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",dreambooth_state_dict)

            noise_scheduler_kwargs = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "linear",
                'clip_sample':   False,
                'steps_offset': 1
            }
            #Determine the scheduler class based on the scheduler variable
            SchedulerClass = DDPMScheduler if scheduler == "DDPMScheduler" else DDIMScheduler
            print(f"using {SchedulerClass.__name__} for training")

            # Set the beta_schedule and create the default noise scheduler
            noise_scheduler_kwargs['beta_schedule'] = 'linear'
            noise_scheduler = SchedulerClass(**noise_scheduler_kwargs)

            # Set the beta_schedule for the spatial noise scheduler and create it
            noise_scheduler_kwargs['beta_schedule'] = 'scaled_linear'
            train_noise_scheduler_spatial = SchedulerClass(**noise_scheduler_kwargs)

            # Reset the beta_schedule for the linear noise scheduler and create it
            noise_scheduler_kwargs['beta_schedule'] = 'linear'
            train_noise_scheduler = SchedulerClass(**noise_scheduler_kwargs)

            # 1. vae
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(dreambooth_state_dict, converted_vae_config)
            vae = AutoencoderKL(**converted_vae_config)
            vae.load_state_dict(converted_vae, strict=False)

            # 2. unet
            converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
            converted_unet = convert_ldm_unet_checkpoint(dreambooth_state_dict, converted_unet_config)
            unet = UNet3DConditionModel(**ad_unet_config)       # type: ignore
            unet.load_state_dict(converted_unet, strict=False)

            del dreambooth_state_dict, converted_unet, converted_vae

            # Validation pipeline
            validation_pipeline = AnimationPipeline(
                unet=unet, 
                vae=vae, 
                tokenizer=tokenizer, 
                text_encoder=text_encoder, 
                scheduler=noise_scheduler,      # type: ignore
            )
            # Freeze all models for LoRA training
            unet.requires_grad_(False)
            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)

            #xformers
            if XFORMERS_IS_AVAILABLE:
                if use_xformers:
                    unet.enable_xformers_memory_efficient_attention()
                else:
                    unet.disable_xformers_memory_efficient_attention()

            # Enable gradient checkpointing
            unet.enable_gradient_checkpointing()

            motion_model, domain_adapter_path = additional_models 

            validation_pipeline = load_weights(
                validation_pipeline, 
                motion_model=motion_model,
                adapter_lora_path=domain_adapter_path, 
                dreambooth_model_path=""
            )

            validation_pipeline.enable_vae_slicing()         

            pipeline = _Pipeline(
                validation_pipeline=validation_pipeline,
                train_noise_scheduler=train_noise_scheduler,
                train_noise_scheduler_spatial=train_noise_scheduler_spatial,
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer
            )

            return (pipeline,)
        
class ADMD_ComfyModelLoader:
    @classmethod
    def IS_CHANGED(cls):
        return ""
    @classmethod
    def INPUT_TYPES(cls):

        return {"required":
                {
                "model" :("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "motion_model":("MOTION_MODEL_ADE",),
                "scheduler": (
            [   
                'DDIMScheduler',
                'DDPMScheduler',
            ], {
               "default": 'DDIMScheduler'
            }),
            "use_xformers": ("BOOLEAN", {"default": False}),
                },                
            }
    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "load_checkpoint"

    CATEGORY = "AD_MotionDirector"

    def load_checkpoint(self, model, clip, vae, scheduler, use_xformers, motion_model):
        with torch.inference_mode(False):
        
            pbar = comfy.utils.ProgressBar(4)
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
            ad_unet_config = OmegaConf.load(os.path.join(script_directory, f"configs/ad_unet_config.yaml"))

            from diffusers.loaders.single_file_utils import (convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint, create_text_encoder_from_ldm_clip_checkpoint, create_vae_diffusers_config, create_unet_diffusers_config)

            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

            clip_sd = None
            load_models = [model]
            load_models.append(clip.load_model())
            clip_sd = clip.get_sd()
            
            comfy.model_management.load_models_gpu(load_models)
            sd = model.model.state_dict_for_saving(clip_sd, vae.get_sd(), None)
     
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",sd)
            pbar.update(1)
            noise_scheduler_kwargs = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "linear",
                'clip_sample':   False,
                'steps_offset': 1
            }
            #Determine the scheduler class based on the scheduler variable
            SchedulerClass = DDPMScheduler if scheduler == "DDPMScheduler" else DDIMScheduler
            print(f"using {SchedulerClass.__name__} for training")

            # Set the beta_schedule and create the default noise scheduler
            noise_scheduler_kwargs['beta_schedule'] = 'linear'
            noise_scheduler = SchedulerClass(**noise_scheduler_kwargs)

            # Set the beta_schedule for the spatial noise scheduler and create it
            noise_scheduler_kwargs['beta_schedule'] = 'scaled_linear'
            train_noise_scheduler_spatial = SchedulerClass(**noise_scheduler_kwargs)

            # Reset the beta_schedule for the linear noise scheduler and create it
            noise_scheduler_kwargs['beta_schedule'] = 'linear'
            train_noise_scheduler = SchedulerClass(**noise_scheduler_kwargs)

            # 1. vae
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            vae = AutoencoderKL(**converted_vae_config)
            vae.load_state_dict(converted_vae, strict=False)
            pbar.update(1)
            # 2. unet
            converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            unet = UNet3DConditionModel(**ad_unet_config)       # type: ignore
            unet.load_state_dict(converted_unet, strict=False)
            
            del sd, converted_unet, converted_vae

            pbar.update(1)

            # Validation pipeline
            validation_pipeline = AnimationPipeline(
                unet=unet, 
                vae=vae, 
                tokenizer=tokenizer, 
                text_encoder=text_encoder,
                scheduler=noise_scheduler, # type: ignore
            )
            # Freeze all models for LoRA training
            unet.requires_grad_(False)
            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)

            #xformers
            if XFORMERS_IS_AVAILABLE:
                if use_xformers:
                    unet.enable_xformers_memory_efficient_attention()
                else:
                    unet.disable_xformers_memory_efficient_attention()

            # Enable gradient checkpointing
            unet.enable_gradient_checkpointing()

            domain_adapter_path = ""

            validation_pipeline = load_weights(
                validation_pipeline, 
                motion_model=motion_model,
                adapter_lora_path=domain_adapter_path, 
                dreambooth_model_path=""
            )
            pbar.update(1)
            validation_pipeline.enable_vae_slicing()         


            pipeline = _Pipeline(
                validation_pipeline=validation_pipeline,
                train_noise_scheduler=train_noise_scheduler,
                train_noise_scheduler_spatial=train_noise_scheduler_spatial,
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer
            )

            return (pipeline,)
        
class ADMD_AdditionalModelSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "motion_module": (folder_paths.get_filename_list("animatediff_models"),),
                "use_adapter_lora": ("BOOLEAN", {"default": True}),                                       
            },
            "optional": {
                "optional_adapter_lora": (folder_paths.get_filename_list("loras"),),       
            }
        }
    RETURN_TYPES = ("ADDITIONAL_MODELS",)
    RETURN_NAMES = ("additional_models",)
    FUNCTION = "select_models"

    CATEGORY = "AD_MotionDirector"

    def select_models(self, motion_module, use_adapter_lora, optional_adapter_lora=""):
        additional_models = []
        motion_model = folder_paths.get_full_path("animatediff_models", motion_module)
        if not Path(motion_model).is_file():                # type: ignore
            raise ValueError(f"Motion model {motion_model} does not exist")
        if use_adapter_lora:
            adapter_lora_path = folder_paths.get_full_path("loras", optional_adapter_lora)
            if not Path(adapter_lora_path).is_file():       # type: ignore
                raise ValueError(f"Adapter LoRA path {adapter_lora_path} does not exist")
        else:
            adapter_lora_path = ""        

        additional_models.append(motion_model)
        additional_models.append(adapter_lora_path)  
        return (additional_models,)

class ADMD_ValidationSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "inference_steps": ("INT", {"default": 25, "min": 0, "max": 256, "step": 1}),
        "guidance_scale": ("FLOAT", {"default": 8, "min": 0, "max": 32, "step": 0.1}),
        "spatial_scale": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),                                        
        "validation_prompt": ("STRING", {"multiline": True, "default": "",}),
        },
          
        }
    RETURN_TYPES = ("VALIDATION_SETTINGS",)
    RETURN_NAMES = ("validation_settings",)
    FUNCTION = "create_validation_settings"

    CATEGORY = "AD_MotionDirector"

    def create_validation_settings(self, inference_steps, guidance_scale, spatial_scale, seed, validation_prompt):
        # Create a dictionary with the local variables
        local_vars = locals()
        
        # Filter the dictionary to include only the variables you want
        validation_settings = {
            "inference_steps": local_vars["inference_steps"],
            "guidance_scale": local_vars["guidance_scale"],
            "spatial_scale": local_vars["spatial_scale"],
            "seed": local_vars["seed"],
            "validation_prompt": local_vars["validation_prompt"]
        }
       
        return validation_settings,

class ADMD_LoadLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {"forceInput": True, "multiline": False, "default": "",}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "prev_motion_lora": ("MOTION_LORA",),
            }
        }
    
    RETURN_TYPES = ("MOTION_LORA",)
    CATEGORY = "AD_MotionDirector"
    FUNCTION = "load_motion_lora"

    def load_motion_lora(self, lora_path: str, strength: float, prev_motion_lora: Union[MotionLoraList, None]=None):
        
        if prev_motion_lora is None:
            prev_motion_lora = MotionLoraList()
        else:
            prev_motion_lora = prev_motion_lora.clone()
        full_lora_path = os.path.join(folder_paths.models_dir,"animatediff_motion_lora",lora_path)
        # check if motion lora with name exists
        if not Path(full_lora_path).is_file():
            raise FileNotFoundError(f"Motion lora not found at {full_lora_path}")
        # create motion lora info to be loaded in AnimateDiff Loader
        lora_info = MotionLoraInfo(name=lora_path, strength=strength)
        prev_motion_lora.add_lora(lora_info)

        return (prev_motion_lora,)

class ADMD_SaveLora:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "admd_pipeline": ("ADMDPIPELINE", ),
                "lora_info": ("LORAINFO", ),
            },
        }
    
    RETURN_TYPES = ("STRING", "ADMDPIPELINE",)
    RETURN_NAMES = ("lora_path", "admd_pipeline",)
    CATEGORY = "AD_MotionDirector"
    FUNCTION = "save_motion_lora"

    def save_motion_lora(self, admd_pipeline: _ADMD_Pipeline, lora_info):
        with torch.inference_mode(False):
            validation_pipeline = admd_pipeline.validation_pipeline
            global_step = admd_pipeline.global_step

            import copy
            validation_pipeline.to('cpu')  # We do this to prevent VRAM spiking / increase from the new copy
            
            spatial_lora_path = lora_info['spatial_lora_path']
            temporal_lora_path = lora_info['temporal_lora_path']
            lora_name = lora_info['lora_name']
            lora_rank = lora_info['lora_rank']
            temporal_lora_base_path = lora_info['temporal_lora_base_path']

            lora_manager_spatial = LoraHandler(use_unet_lora=True, unet_replace_modules=["Transformer3DModel"])
            lora_manager_spatial.save_lora_weights(
                model=copy.deepcopy(validation_pipeline),       # type: ignore
                save_path=spatial_lora_path, 
                step=str(global_step),
                use_safetensors=True,
                lora_rank=lora_rank,
                lora_name=lora_name + "_r"+ str(lora_rank) + "_spatial",
            )
            lora_manager_temporal = LoraHandler(use_unet_lora=True, unet_replace_modules=["TemporalTransformerBlock"])
            if lora_manager_temporal is not None:
                lora_manager_temporal.save_lora_weights(
                    model=copy.deepcopy(validation_pipeline),   # type: ignore
                    save_path=temporal_lora_path, 
                    step=str(global_step),
                    use_safetensors=True,
                    lora_rank=lora_rank,
                    lora_name=lora_name + "_r"+ str(lora_rank) + "_temporal",
                    use_motion_lora_format=True
                )

            final_temporal_lora_name = os.path.join(temporal_lora_base_path, (str(global_step) + "_" + lora_name + "_r"+ str(lora_rank) + "_temporal_unet.safetensors"))
       
            return (final_temporal_lora_name, admd_pipeline)

class ADMD_TrainLora:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "admd_pipeline": ("ADMDPIPELINE", ),
                "steps": ("INT", {"default": 100, "min": 0, "max": 10000, "step": 1}),
            },
            "optional": {
                "trigger_input": ("VHS_FILENAMES", ), #attempt to force comfy execution order
                "opt_images_override": ("IMAGE", ),
            }
        }
    
    RETURN_TYPES = ("ADMDPIPELINE",)
    RETURN_NAMES = ("admd_pipeline",)
    CATEGORY = "AD_MotionDirector"
    FUNCTION = "train"

    def train(self, admd_pipeline: _ADMD_Pipeline, steps, opt_images_override=None, trigger_input=None):
        with torch.inference_mode(False):
            train_noise_scheduler = admd_pipeline.train_noise_scheduler
            train_noise_scheduler_spatial = admd_pipeline.train_noise_scheduler_spatial
            
            text_encoder = admd_pipeline.text_encoder
            tokenizer = admd_pipeline.tokenizer

            input_sched = admd_pipeline.inputs
            
            optimizer_temporal = admd_pipeline.optimizer_temporal
            optimizer_spatial_list = admd_pipeline.optimizer_spatial_list
            lr_scheduler_spatial_list = admd_pipeline.lr_scheduler_spatial_list
            lr_scheduler_temporal = admd_pipeline.lr_scheduler_temporal
            scaler = admd_pipeline.scaler
            seed = admd_pipeline.seed
            include_resnet = admd_pipeline.include_resnet
            use_offset_noise = False

            torch.manual_seed(seed)

            device = typing.cast(torch.device, comfy.model_management.get_torch_device())
            comfy.model_management.unload_all_models()

            unet = admd_pipeline.unet
            vae = admd_pipeline.vae

            unet.to(device)
            vae.to(device)
            text_encoder.to(device) # type: ignore
            unet.enable_gradient_checkpointing()
            unet.train()

            if include_resnet:
                target_spatial_modules = ["Transformer3DModel", "ResnetBlock2D"]
            else: 
                target_spatial_modules = ["Transformer3DModel"]

            target_temporal_modules = ["TemporalTransformerBlock"]
      
            global_step = admd_pipeline.global_step
            print(f"global_step: {global_step}")
            
            inputs: utils.scheduling.InputSchedule[Any, utils.inputs.TrainingVideo]
            # Get the text embedding for conditioning
            with torch.no_grad():
                
                inputs = utils.scheduling.InputSchedule.prepare_dataset(
                    input_sched, 
                    vae=vae, 
                    tokenizer=tokenizer, 
                    text_encoder=text_encoder, device=device
                )

                #num_update_steps_per_epoch = math.ceil(batch_size) / gradient_accumulation_steps
                #num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

                progress_bar = tqdm(range(0, steps))
                progress_bar.set_description("Steps")
                pbar = comfy.utils.ProgressBar(steps)

            ### <<<< Training <<<< ###
            #for epoch in range(first_epoch, steps):
            for video in inputs.train(steps):
                pixel_values = video.pixels
                latents = video.latent
                encoder_hidden_states = video.encoder_hidden_states
                # encoder_hidden_states = next(encoder_hidden_states_cycle)
                
                spatial_scheduler_lr = 0.0
                temporal_scheduler_lr = 0.0

                # Handle Lora Optimizers & Conditions
                for optimizer_spatial in optimizer_spatial_list:
                    optimizer_spatial.zero_grad(set_to_none=True)

                if optimizer_temporal is not None:
                    optimizer_temporal.zero_grad(set_to_none=True)
    
                mask_spatial_lora = random.uniform(0, 1) < 0.2
                #mask_spatial_lora = 0

                # Sample a random timestep for each video
                timesteps = torch.randint(0, 1000, (1,), device=pixel_values.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                                    
                noise = sample_noise(latents, 0, use_offset_noise=use_offset_noise)
                comfy.model_management.soft_empty_cache()
                target = noise

                with torch.cuda.amp.autocast():
                    if mask_spatial_lora:
                        loras = extract_lora_child_module(unet, target_replace_module=target_spatial_modules)
                        scale_loras(loras, 0.)
                        loss_spatial = None
                    else:
                        loras = extract_lora_child_module(unet, target_replace_module=target_spatial_modules)
                        scale_loras(loras, 1.0)

                        loras = extract_lora_child_module(unet, target_replace_module=target_temporal_modules)
                        if len(loras) > 0:
                            scale_loras(loras, 0.)
                        
                        ### >>>> Spatial LoRA Prediction >>>> ###
                        noisy_latents = train_noise_scheduler_spatial.add_noise(latents, noise, timesteps) # type: ignore
                        noisy_latents_input, target_spatial = get_spatial_latents(
                            pixel_values,  
                            noisy_latents,
                            target,
                        )
                        model_pred_spatial = unet(noisy_latents_input.unsqueeze(2), timesteps,
                                                encoder_hidden_states=encoder_hidden_states).sample
                        loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                                target_spatial.float(), reduction="mean")
                        
                    loras = extract_lora_child_module(unet, target_replace_module=target_temporal_modules)
                    scale_loras(loras, 1.0)
                    
                    ### >>>> Temporal LoRA Prediction >>>> ###
                    noisy_latents = train_noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                    
                    loss_temporal = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss_temporal = create_ad_temporal_loss(model_pred, loss_temporal, target)
                        
                    # Backpropagate
                    if not mask_spatial_lora:
                        scaler.scale(loss_spatial).backward(retain_graph=True)          # type: ignore
                        scaler.step(optimizer_spatial_list[0])
                                        
                    scaler.scale(loss_temporal).backward()
                    scaler.step(optimizer_temporal)
        
                    lr_scheduler_spatial_list[0].step()
                    spatial_scheduler_lr = lr_scheduler_spatial_list[0].get_lr()[0]     # type: ignore
                        
                    if lr_scheduler_temporal is not None:
                        lr_scheduler_temporal.step()
                        temporal_scheduler_lr = lr_scheduler_temporal.get_lr()[0]       # type: ignore
            
                scaler.update()
                progress_bar.update(1)
                pbar.update(1)
                global_step += 1
                logs = {
                    "Temporal Loss": loss_temporal.detach().item(),
                    "Temporal LR": temporal_scheduler_lr, 
                    "Spatial Loss": loss_spatial.detach().item() if loss_spatial is not None else 0,
                    "Spatial LR": spatial_scheduler_lr
                }
                progress_bar.set_postfix(**logs)

                # if global_step >= steps:
                #     print("BREAK")
                #     break

            admd_pipeline.global_step = global_step
            admd_pipeline.unet = unet
            admd_pipeline.scaler = scaler

            comfy.model_management.soft_empty_cache()
            return (admd_pipeline,)

class ADMD_ValidationSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "validation_settings": ("VALIDATION_SETTINGS", ),
                "admd_pipeline": ("ADMDPIPELINE", ),
                "width": ("INT", { "default": 512, "min": 0,"step": 8 }),
                "height": ("INT", { "default": 512, "min": 0,"step": 8 }),
                "frames": ("INT", { "default": 16, "min": 0,"step": 1 }),
            },
        }
    
    RETURN_TYPES = ("ADMDPIPELINE", "IMAGE",)
    RETURN_NAMES = ("admd_pipeline", "images",)
    CATEGORY = "AD_MotionDirector"
    FUNCTION = "train"

    def train(self, admd_pipeline: _ADMD_Pipeline, validation_settings, width: int, height: int, frames: int):
        with torch.inference_mode(False):       
            unet = admd_pipeline.unet
            text_encoder = admd_pipeline.text_encoder
            vae = admd_pipeline.vae
            
            validation_pipeline = admd_pipeline.validation_pipeline
          
            device = comfy.model_management.get_torch_device()
            # B, H, W, C = pixel_values[0].shape

            unet.to(device)
            vae.to(device)
            text_encoder.to(device) # type: ignore
            
            validation_inference_steps = validation_settings["inference_steps"]
            validation_guidance_scale = validation_settings["guidance_scale"]
            validation_spatial_scale = validation_settings["spatial_scale"]
            validation_seed = validation_settings["seed"]
            validation_prompt = validation_settings["validation_prompt"]
                      
            with torch.inference_mode(True):
                samples = []
                generator = torch.Generator(device=device)
                generator.manual_seed(validation_seed)
                
                with torch.cuda.amp.autocast(enabled=True):
                    unet.disable_gradient_checkpointing()
                    unet.eval()
                    loras = extract_lora_child_module(
                        unet, 
                        target_replace_module=["Transformer3DModel"]
                    )
                    scale_loras(loras, validation_spatial_scale)
                    
                    with torch.inference_mode(True):
                        prompt = validation_prompt
                        
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = frames,
                            height       = height,
                            width        = width,
                            num_inference_steps = validation_inference_steps,
                            guidance_scale = validation_guidance_scale,
                        ).videos # type: ignore
                        samples.append(sample)
                # Reshape the sample tensor for returning
                samples = torch.concat(samples)
                samples = samples.view(*samples.shape[1:])
                samples = samples.permute(1, 2, 3, 0).cpu() 
                return (admd_pipeline, samples,)
            
class ADMD_MakeBatchList:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "images2": ("IMAGE", ),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch_list",)
    CATEGORY = "AD_MotionDirector"
    FUNCTION = "batchlist"

    def batchlist(self, images, images2):
        batch_list = []
        
        if isinstance(images, list):
            batch_list.extend(images)
        else:
            batch_list.append(images)

        if isinstance(images2, list):
            batch_list.extend(images2)
        else:
            batch_list.append(images2)
        
        return (batch_list,)

NODE_CLASS_MAPPINGS = {
    "ADMD_InitializeTraining": ADMD_InitializeTraining,
    "ADMD_DiffusersLoader": ADMD_DiffusersLoader,
    "ADMD_AdditionalModelSelect": ADMD_AdditionalModelSelect,
    "ADMD_ValidationSettings": ADMD_ValidationSettings,
    "ADMD_LoadLora": ADMD_LoadLora,
    "ADMD_SaveLora": ADMD_SaveLora,
    "ADMD_TrainLora": ADMD_TrainLora,
    "ADMD_CheckpointLoader": ADMD_CheckpointLoader,
    "ADMD_ValidationSampler": ADMD_ValidationSampler,
    "ADMD_MakeBatchList": ADMD_MakeBatchList,
    "ADMD_ComfyModelLoader": ADMD_ComfyModelLoader,
    "ADMD_Ext_InputPipeline": ADMD_Ext_InputPipeline,
    "ADMD_Ext_InputPipeStart": ADMD_Ext_InputPipeStart,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ADMD_InitializeTraining": "ADMD_InitializeTraining",
    "ADMD_DiffusersLoader": "ADMD_DiffusersLoader",
    "ADMD_AdditionalModelSelect": "ADMD_AdditionalModelSelect",
    "ADMD_ValidationSettings": "ADMD_ValidationSettings",
    "ADMD_LoadLora": "ADMD_LoadLora",
    "ADMD_SaveLora": "ADMD_SaveLora",
    "ADMD_TrainLora": "ADMD_TrainLora",
    "ADMD_CheckpointLoader": "ADMD_CheckpointLoader",
    "ADMD_ValidationSampler": "ADMD_ValidationSampler",
    "ADMD_MakeBatchList": "ADMD_MakeBatchList",
    "ADMD_ComfyModelLoader": "ADMD_ComfyModelLoader",
    "ADMD_Ext_InputPipeline": "ADMD_Ext_InputPipeline",
    "ADMD_Ext_InputPipeStart": "ADMD_Ext_InputPipeStart",
}