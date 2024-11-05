from contextlib import nullcontext
from enum import StrEnum
import os
import csv
import copy
from typing import Any, Callable, Dict, Iterator, List, Set, ContextManager, Tuple, Type

import timm
import timm.optim
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, CLIPModel
from diffusers import StableDiffusionPipeline
from diffusers import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer 
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)
import torch
from torch import nn, optim
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributed._composable import checkpoint


DEVICE = "cuda:0"
BASE_DIR = "/n/holyscratch01/idreos_lab/Users/spurandare/auto-sac"
# BASE_DIR = "/n/holylabs/LABS/acc_lab/Users/golden/CS2881r/mem-run-estimator-fork"
OUT_DIR = f"{BASE_DIR}/outputs"
gpu_types: Set[str] = {"H100", "A100"}

runtime_est_modes: Set[str] = {"operator-level-cost-model", "operator-level-benchmark", "operator-level-learned-model"}

model_names: Set[str] = {
    "hf_T5",
    "hf_GPT2",
    "timm_vit",
    "hf_clip",
    "llama_v3_1b",
    "gemma_2b",
    "timm_convnext_v2",
    "stable_diffusion",
    "flux"
}

class ExpType(StrEnum):
    runtime_est = "runtime_estimation"
    memory_est = "memory_estimation"
    real_execution = "real_execution"
    test = "test"

class Precision(StrEnum):
    FP = "FP"
    MP = "MP"
    HP = "HP"

model_cards: Dict[str, str] = {
    "hf_T5": "t5-large",
    "hf_GPT2": "gpt2-large",
    "llama_v3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma_2b": "google/gemma-2b",
    "timm_convnext_v2": "convnextv2_huge.fcmae_ft_in22k_in1k_512",
    "timm_vit": "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",
    "hf_clip": "openai/clip-vit-large-patch14-336",
    "stable_diffusion": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "flux": "black-forest-labs/FLUX.1-schnell",
}

precision_to_dtype: Dict[Precision, torch.dtype] = {
    Precision.FP : torch.float32,
    Precision.HP: torch.float16,
    Precision.MP: torch.float32
}

model_class: Dict[str, Type] = {
    "hf_T5": AutoModelForSeq2SeqLM,
    "hf_GPT2": AutoModelForCausalLM,
    "llama_v3_1b": AutoModelForCausalLM,
    "gemma_2b": AutoModelForCausalLM,
    "hf_clip": CLIPModel,
    "stable_diffusion": [UNet2DConditionModel, AutoencoderKL, CLIPTextModel, DDPMScheduler],
    "flux": FluxTransformer2DModel
}

model_ac_classes: Dict[str, List[str]] = {
    "hf_T5": ["T5LayerFF", "T5LayerNorm"],
    "hf_GPT2": ["GPT2Block",],
    "llama_v3_1b": ["LlamaMLP","LlamaRMSNorm"],
    "gemma_2b": ["GemmaMLP", "GemmaRMSNorm"],
    "timm_convnext_v2": ["GlobalResponseNormMlp",],
    "timm_vit": ["Block",],
    "hf_clip": ["CLIPEncoderLayer",],
    "stable_diffusion": ["CrossAttnDownBlock2D", "DownBlock2D", "UpBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn"], ### ???
    "flux": ["FluxTransformer2DModel"] ### ???
}

def generate_inputs_and_labels(
        bsz: int, vocab_size: int, seq_len: int, dev: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len), dtype=torch.int64, device=dev)
    labels = torch.randint(0, vocab_size, (bsz, seq_len), dtype=torch.int64, device=dev)
    return (input_ids, labels)

def generate_inputs_and_targets(
        bsz: int, im_sz:int, n_classes: int, dtype: torch.dtype, dev: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    input = torch.randn((bsz, 3, im_sz, im_sz), dtype=dtype, device=dev)
    target = torch.randint(0, n_classes, (bsz, ), dtype=torch.int64, device=dev)
    return(input, target)

def generate_multimodal_inputs(
        bsz: int, vocab_size: int, seq_len: int, im_sz:int, dtype: torch.dtype, dev: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_img = torch.randn((bsz, 3, im_sz, im_sz), dtype=dtype, device=dev)
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len), dtype=torch.int64, device=dev)
    attention_mask = torch.ones((bsz, seq_len), dtype=torch.int64, device=dev)
    return (input_img, input_ids, attention_mask)
     
def generate_noise_and_timesteps(
        bsz: int, im_sz:int, num_denoise:int, dtype: torch.dtype, dev: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    pixel_img = torch.randn(bsz, 3, im_sz, im_sz, dtype=dtype, device=dev)
    timesteps = torch.randint(0, num_denoise, (bsz,), dtype=torch.int64, device=dev)
    return (pixel_img, timesteps)

def generate_flux_inputs(
        batch_size: int, image_size:int, seq_len:int, in_channels:int, joint_attention_dim:int, pooled_projection_dim:int, dtype: torch.dtype, dev: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    hidden_states = torch.randn((batch_size, int((image_size/8)*2), in_channels), dtype=dtype, device=dev)
    encoder_hidden_states = torch.randn((batch_size, seq_len, joint_attention_dim), dtype=dtype, device=dev)
    pooled_projections = torch.randn((batch_size, pooled_projection_dim), dtype=dtype, device=dev)
    timestep = torch.tensor([0], dtype=dtype, device=dev)
    img_ids = torch.randn((int((image_size/8)*2), 3), dtype=dtype, device=dev)
    txt_ids = torch.randn((seq_len, 3), dtype=dtype, device=dev)
    target = torch.randn_like(hidden_states)

    return (hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, target)



def create_optimizer(param_iter: Iterator) -> optim.Optimizer:
    optimizer = optim.Adam(
        param_iter,
        lr=1e-4,
        weight_decay=1.0e-4,
        eps=1.0e-6,
    )
    return optimizer

def apply_ac(model: nn.Module, ac_classes: List[str]):
    for module in model.modules():
        module_class = module.__class__.__name__
        if module_class in ac_classes:
            checkpoint(module, preserve_rng_state=False)

def create_training_setup(
        model_name: str,
        batch_size: int = 2,
        seq_len: int = 128,
        precision: Precision = Precision.HP,
        ac: bool = False,
        image_size: int = 224,
        dev: torch.device = torch.device(DEVICE), 
        init_mode: ContextManager = nullcontext(),
        num_denoising_steps: int=50,
    ) -> Tuple[List[nn.Module], optim.Optimizer, Callable]:
    dtype = precision_to_dtype[precision]
    amp_context = nullcontext()
    if precision == Precision.MP:
        amp_context = torch.autocast(device_type=DEVICE)
    if model_name in [
        "hf_T5", "hf_GPT2", "llama_v3_1b", "gemma_2b"
    ]:  
        
        model_card = model_cards[model_name]
        model_cls = model_class[model_name]
        config = AutoConfig.from_pretrained(model_card)

        with init_mode:
            with torch.device(dev):
                model = model_cls.from_config(config=config).to(dtype=dtype)
            optimizer = create_optimizer(model.parameters())
            if ac:
                ac_classes = model_ac_classes[model_name]
                apply_ac(model, ac_classes)

        def hf_train_step(
                models: List[nn.Module], optim: optim.Optimizer,
            ):
                model = models[0]
                input_ids, labels = generate_inputs_and_labels(batch_size, config.vocab_size, seq_len, dev)
                inputs = {"input_ids": input_ids, "labels": labels}
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        loss = model(**inputs).loss
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

        return ([model], optimizer, hf_train_step)

    elif model_name in ["timm_vit", "timm_convnext_v2"]:
        model_card = model_cards[model_name]
        with init_mode:
            with torch.device(dev):
                model = timm.create_model(model_card, pretrained=False).to(dtype=dtype)
            optimizer = timm.optim.create_optimizer_v2(model, opt="adam")     
            loss_fn = nn.functional.cross_entropy
            if ac:
                ac_classes = model_ac_classes[model_name]
                apply_ac(model, ac_classes)

        def timm_train_step(
                models: List[nn.Module], optim: optim.Optimizer,
            ):
                model = models[0]
                n_classes = model.default_cfg['num_classes']
                inputs = generate_inputs_and_targets(batch_size, image_size, n_classes, dtype, dev)
                inp, target = inputs
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        output = model(inp)
                        loss = loss_fn(output, target)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

        return ([model], optimizer, timm_train_step)
    
    elif model_name == "hf_clip":
        model_card = model_cards[model_name]
        model_cls = model_class[model_name]
        config = AutoConfig.from_pretrained(model_card)
        with init_mode:
            with torch.device(dev):
                model = model_cls._from_config(config=config).to(dtype=dtype)
                loss_fn = ContrastiveLossWithTemperature()

            class CLIP(nn.Module):
                def __init__(self, clip_model, loss_mod):
                    super().__init__()
                    self.add_module('clip_model', clip_model)
                    self.add_module('contrastive_loss_with_temp', loss_mod)

                def forward(self, **kwargs):
                    outputs = self.clip_model(**kwargs)
                    loss = self.contrastive_loss_with_temp(outputs.image_embeds, outputs.text_embeds)
                    return loss

            model_with_loss = CLIP(model, loss_fn)
            if ac:
                ac_classes = model_ac_classes[model_name]
                apply_ac(model_with_loss, ac_classes)
            optimizer = create_optimizer(model_with_loss.parameters())
        
        def clip_train_step(
            models: List[nn.Module], optim: optim.Optimizer,
        ):
                model = models[0]
                img, ids, attn_mask = generate_multimodal_inputs(
                    batch_size,
                    model.clip_model.config.text_config.vocab_size,
                    model.clip_model.config.text_config.max_length,
                    image_size,
                    dtype,
                    dev
                )
                inputs = {'input_ids': ids, 'attention_mask': attn_mask, 'pixel_values': img}
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        loss = model(**inputs)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

        return ([model_with_loss], optimizer, clip_train_step)

    elif model_name == "stable_diffusion":
        
        model_card = model_cards[model_name]
        model_cls = model_class[model_name]
        unet_config = model_cls[0].load_config(model_card, subfolder="unet")
        vae_config = model_cls[1].load_config(model_card, subfolder="vae")
        text_encoder_config = AutoConfig.from_pretrained(model_card, subfolder="text_encoder")
        scheduler_config = model_cls[3].load_config(model_card, subfolder="scheduler")

        with init_mode:
            with torch.device(dev):
                unet = model_cls[0].from_config(unet_config).to(dtype=dtype)
                vae = model_cls[1].from_config(vae_config).to(dtype=dtype)
                text_encoder = model_cls[2]._from_config(text_encoder_config).to(dtype=dtype)
                scheduler = model_cls[3].from_config(scheduler_config)
                print(type(scheduler))

            optimizer = create_optimizer(unet.parameters())
            

            if ac:
                ac_classes = model_ac_classes[model_name]
                apply_ac(unet, ac_classes)


        def sd_train_step(
                models: List[nn.Module], optim: optim.Optimizer,
            ):
                # Generate text embeddings
                unet, text_encoder, vae = models[0], models[1], models[2]
                input_ids, _ = generate_inputs_and_labels(batch_size, text_encoder.config.vocab_size, seq_len, dev)
                text_embeddings = text_encoder(input_ids).last_hidden_state
                text_embeddings = text_embeddings

                # Generate timesteps and pixel_image
                pixel_img, timesteps = generate_noise_and_timesteps(batch_size, image_size, num_denoising_steps, dtype, dev)

                # Generate noisy latents
                with torch.no_grad():
                    latents = vae.encode(pixel_img).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = noisy_latents

                # Prepare inputs for UNet
                inputs = {
                    "encoder_hidden_states": text_embeddings,  # Text embeddings as conditioning information
                    "timestep": timesteps,    # Current timestep
                    "sample": noisy_latents  # Noisy latents as input
                }

                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        noise_pred = unet(**inputs).sample
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

        return ([unet, text_encoder, vae] , optimizer, sd_train_step)


    elif model_name == "flux":

        model_card = model_cards[model_name]
        model_cls = model_class[model_name]

        with init_mode:
            with torch.device(dev):
                transformer = model_cls.from_pretrained(model_card, subfolder="transformer", torch_dtype=dtype)

            optimizer = create_optimizer(transformer.parameters())

            if ac:
                ac_classes = model_ac_classes[model_name]
                apply_ac(unet, ac_classes)

        # Training loop
        def flux_train_step(
            models: List[nn.Module], optim: optim.Optimizer,
        ):
            
            model = models[0]
            # Example inputs for training (replace with actual data as needed)
            in_channels = model.in_channels
            joint_attention_dim = model.joint_attention_dim
            pooled_projection_dim = model.pooled_projection_dim

            hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, target = generate_flux_inputs(batch_size, image_size, seq_len, in_channels, joint_attention_dim, pooled_projection_dim, dtype, dev)
            
            # Prepare inputs for FluxTransformer
            inputs = {
                "hidden_states": hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_projections": pooled_projections,
                "timestep": timestep,
                "img_ids": img_ids,
                "txt_ids": txt_ids,
            }

            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                with amp_context:
                    output = transformer(**inputs).sample
                    loss = nn.functional.mse_loss(output, target) 

                loss.backward()
                optim.step()
                optim.zero_grad()

        return ([transformer], optimizer, flux_train_step)

    else:
         raise ValueError(f"No setup is available for {model_name}. Please choose from {model_names}")

import fcntl

def write_to_logfile(file_name: str, log_record: str):
    with open(file_name, 'a', newline='') as csvfile:
        fcntl.lockf(csvfile, fcntl.LOCK_EX)
        writer = csv.writer(csvfile)
        writer.writerow(log_record)
        fcntl.lockf(csvfile, fcntl.LOCK_UN)

def override_args_with_configs(args, config: Dict[str, Any]):
    b_args = copy.deepcopy(args)
    b_args.batch_size = config["batch_size"]
    b_args.seq_len = config["seq_len"]
    b_args.precision = config["precision"].value
    b_args.enable_ac = config["ac"]
    b_args.image_size = config["image_size"]
    b_args.num_denoising_steps = config["num_denoising_steps"]
    return b_args