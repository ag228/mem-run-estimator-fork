from contextlib import nullcontext
from enum import StrEnum
import csv
import copy
import fcntl
from typing import Any, Callable, Dict, Iterator, List, Set, ContextManager, Tuple, Type, Union

import timm
import timm.optim
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, CLIPModel
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer 
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)
import torch
from torch import nn, optim
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributed.fsdp._wrap_utils import _post_order_apply
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


DEVICE = "cuda:0"
BASE_DIR = "/n/netscratch/idreos_lab/Lab/spurandare/auto-sac"
# BASE_DIR = "/n/holylabs/LABS/acc_lab/Users/golden/CS2881r/mem-run-estimator-fork"
OUT_DIR = f"{BASE_DIR}/outputs"
gpu_types: Set[str] = {"H100", "A100"}

runtime_est_modes: Set[str] = {"operator-level-cost-model", "operator-level-benchmark", "operator-level-learned-model"}

model_names: Set[str] = {
    "hf_T5",
    "timm_vit",
    "hf_clip",
    "llama_v3_1b",
    "gemma_2b",
    "timm_convnext_v2",
    "stable_diffusion",
    "flux",
    "stable_diffusion_mmdit"
}

class ExpType(StrEnum):
    runtime_est = "runtime_estimation"
    memory_est = "memory_estimation"
    real_execution = "real_execution"
    auto_sac = "auto_sac"
    test = "test"

class Precision(StrEnum):
    FP = "FP"
    MP = "MP"
    HP = "HP"

class AC(StrEnum):
    AUTO = "auto"
    FULL = "full"
    NONE = "none"

model_cards: Dict[str, str] = {
    "hf_T5": "t5-large",
    "llama_v3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma_2b": "google/gemma-2b",
    "timm_convnext_v2": "convnextv2_huge.fcmae_ft_in22k_in1k_512",
    "timm_vit": "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",
    "hf_clip": "openai/clip-vit-large-patch14-336",
    "stable_diffusion": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "flux": "black-forest-labs/FLUX.1-schnell",
    "stable_diffusion_mmdit": ["stabilityai/stable-diffusion-3-medium-diffusers"]
}

precision_to_dtype: Dict[Precision, torch.dtype] = {
    Precision.FP : torch.float32,
    Precision.HP: torch.float16,
    Precision.MP: torch.float32
}

model_class: Dict[str, Type] = {
    "hf_T5": AutoModelForSeq2SeqLM,
    "llama_v3_1b": AutoModelForCausalLM,
    "gemma_2b": AutoModelForCausalLM,
    "hf_clip": CLIPModel,
    "stable_diffusion": [UNet2DConditionModel, AutoencoderKL, CLIPTextModel, DDPMScheduler],
    "flux": FluxTransformer2DModel,
    "stable_diffusion_mmdit": [StableDiffusion3Pipeline]
}

model_ac_classes: Dict[str, List[str]] = {
    "hf_T5": ["T5LayerFF", "T5LayerNorm"],
    "llama_v3_1b": ["LlamaDecoderLayer"],
    "gemma_2b": ["GemmaDecoderLayer"],
    "timm_convnext_v2": ["GlobalResponseNormMlp",],
    "timm_vit": ["Block",],
    "hf_clip": ["CLIPEncoderLayer",],
    "stable_diffusion": ["BasicTransformerBlock", "ResnetBlock2D"],
    "flux": ["FluxTransformer2DModel"], ### ???
    "stable_diffusion_mmdit": ["JointTransformerBlock"] ### ???
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

def generate_sd3_inputs(
        batch_size: int, image_size:int, seq_len:int, in_channels:int, joint_attention_dim:int, pooled_projection_dim:int, embed_dim:int, dtype: torch.dtype, dev: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    hidden_states = torch.randn((batch_size, in_channels, image_size, image_size), dtype=dtype, requires_grad=True).to(dev)
    encoder_hidden_states = torch.randn((batch_size, seq_len, embed_dim), dtype=dtype, requires_grad=True).to(dev)
    pooled_projections = torch.randn((batch_size, pooled_projection_dim), dtype=dtype, requires_grad=True).to(dev)
    timestep = torch.tensor([0], dtype=dtype, requires_grad=True).to(dev)
    target = torch.randn_like(hidden_states, requires_grad=True).to(dev)

    return (hidden_states, encoder_hidden_states, pooled_projections, timestep, target)

def create_optimizer(param_iter: Iterator) -> optim.Optimizer:
    optimizer = optim.Adam(
        param_iter,
        lr=1e-4,
        weight_decay=1.0e-4,
        eps=1.0e-6,
    )
    return optimizer

def apply_ac(model: nn.Module, ac_classes: List[str]):
    def ac_wrapper(module: nn.Module) -> Union[nn.Module, None]:
        module_class = module.__class__.__name__
        if module_class in ac_classes:
            return checkpoint_wrapper(
                module,
                preserve_rng_state=False,
            )
        else:
            return None
    _post_order_apply(model, fn=ac_wrapper)

def create_training_setup(
        model_name: str,
        batch_size: int = 2,
        seq_len: int = 128,
        precision: Precision = Precision.HP,
        ac: AC = AC.NONE,
        image_size: int = 224,
        dev: torch.device = torch.device(DEVICE), 
        init_mode: ContextManager = nullcontext(),
        num_denoising_steps: int=50,
    ) -> Tuple[Callable, List[nn.Module], List[optim.Optimizer], Any]:
    dtype = precision_to_dtype[precision]
    amp_context = nullcontext()
    if precision == Precision.MP:
        amp_context = torch.autocast(device_type=DEVICE)
    if model_name in [
        "hf_T5", "llama_v3_1b", "gemma_2b"
    ]:  
        
        model_card = model_cards[model_name]
        model_cls = model_class[model_name]
        config = AutoConfig.from_pretrained(model_card)
        if hasattr(config, "use_cache"):
            setattr(config, "use_cache", False)

        with init_mode:
            with torch.device(dev):
                model = model_cls.from_config(config=config).to(dtype=dtype)
            optimizer = create_optimizer(model.parameters())
            if ac == AC.FULL:
                ac_classes = model_ac_classes[model_name]
                apply_ac(model, ac_classes)
            input_ids, labels = generate_inputs_and_labels(batch_size, config.vocab_size, seq_len, dev)
            inputs = {"input_ids": input_ids, "labels": labels}

        def hf_train_step(
                models: List[nn.Module], optimizers: List[optim.Optimizer], inputs
            ):
                model = models[0]
                optimizer = optimizers[0]
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        loss = model(**inputs).loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        return (hf_train_step, [model], [optimizer], inputs)

    elif model_name in ["timm_vit", "timm_convnext_v2"]:
        model_card = model_cards[model_name]
        with init_mode:
            with torch.device(dev):
                model = timm.create_model(model_card, pretrained=False).to(dtype=dtype)
            optimizer = timm.optim.create_optimizer_v2(model, opt="adam")     
            loss_fn = nn.functional.cross_entropy
            if ac == AC.FULL:
                ac_classes = model_ac_classes[model_name]
                apply_ac(model, ac_classes)
            n_classes = model.default_cfg['num_classes']
            inputs = generate_inputs_and_targets(batch_size, image_size, n_classes, dtype, dev)

        def timm_train_step(
                models: List[nn.Module], optimizers: List[optim.Optimizer], inputs
            ):
                model = models[0]
                optimizer = optimizers[0]
                inp, target = inputs
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        output = model(inp)
                        loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        return (timm_train_step, [model], [optimizer], inputs)
    
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
            if ac == AC.FULL:
                ac_classes = model_ac_classes[model_name]
                apply_ac(model_with_loss, ac_classes)
            optimizer = create_optimizer(model_with_loss.parameters())
            inputs = generate_multimodal_inputs(
                batch_size,
                model.clip_model.config.text_config.vocab_size,
                model.clip_model.config.text_config.max_length,
                image_size,
                dtype,
                dev
            )
        
        def clip_train_step(
            models: List[nn.Module], optimizers: List[optim.Optimizer], inputs
        ):
                model = models[0]
                optimizer = optimizers[0]
                img, ids, attn_mask = inputs
                inputs = {'input_ids': ids, 'attention_mask': attn_mask, 'pixel_values': img}
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        loss = model(**inputs)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        return (clip_train_step, [model_with_loss], [optimizer], inputs)

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
                del vae.decoder

            optimizer = create_optimizer(unet.parameters())
            if ac == AC.FULL:
                ac_classes = model_ac_classes[model_name]
                apply_ac(unet, ac_classes)
            # Generate timesteps and pixel_image
            pixel_img, timesteps = generate_noise_and_timesteps(batch_size, image_size, num_denoising_steps, dtype, dev)
            input_ids, labels = generate_inputs_and_labels(batch_size, text_encoder.config.vocab_size, seq_len, dev)
            inputs = (input_ids, labels, pixel_img, timesteps)

        def sd_train_step(
                models: List[nn.Module], optimizers: List[optim.Optimizer], inputs
            ):
                
                unet, text_encoder, vae_encoder = models[0], models[1], models[2]
                optimizer = optimizers[0]
                input_ids, _, pixel_img, timesteps = inputs
                # Generate text embeddings and noisy latents
                with torch.no_grad():
                    text_embeddings = text_encoder(input_ids).last_hidden_state
                    latents = vae.encode(pixel_img).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Prepare inputs for UNet
                unet_inputs = {
                    "encoder_hidden_states": text_embeddings,
                    "timestep": timesteps, 
                    "sample": noisy_latents  
                }

                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        noise_pred = unet(**unet_inputs).sample
                        loss = torch.nn.functional.mse_loss(noise_pred, latents)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        return (sd_train_step, [unet, text_encoder, vae.encoder], [optimizer], inputs)


    elif model_name == "flux":

        model_card = model_cards[model_name]
        model_cls = model_class[model_name]

        with init_mode:
            with torch.device(dev):
                transformer = model_cls.from_pretrained(model_card, subfolder="transformer", torch_dtype=dtype)

                vae = model_cls.from_pretrained(model_card, subfolder="vae", torch_dtype=dtype)
                text_encoder = model_cls.from_pretrained(model_card, subfolder="text_encoder", torch_dtype=dtype)
                text_encoder_2 = model_cls.from_pretrained(model_card, subfolder="text_encoder_2", torch_dtype=dtype)
                tokenizer = model_cls.from_pretrained(model_card, subfolder="tokenizer", torch_dtype=dtype)
                tokenizer_2 = model_cls.from_pretrained(model_card, subfolder="tokenizer_2", torch_dtype=dtype)
                scheduler = model_cls.from_pretrained(model_card, subfolder="scheduler", torch_dtype=dtype)

            optimizer = create_optimizer(transformer.parameters())

            if ac == AC.FULL:
                ac_classes = model_ac_classes[model_name]
                apply_ac(unet, ac_classes)
            in_channels = model.in_channels
            joint_attention_dim = model.joint_attention_dim
            pooled_projection_dim = model.pooled_projection_dim
            flux_inputs = generate_flux_inputs(
                batch_size, image_size, seq_len, in_channels, joint_attention_dim, pooled_projection_dim, dtype, dev
            )



        # Training loop
        def flux_train_step(
            models: List[nn.Module], optimizers: List[optim.Optimizer], flux_inputs
        ):
            transformer = models[0]
            optimizer = optimizers[0]
            hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, target = flux_inputs


            # overwrite encoder_hidden_states and pooled_projections from real models:

            ### Text Encoder 2
            text_input_ids = torch.randn((batch_size, 77))   # hardcode to 77 for now, todo: add two sequence lengths for flux
            pooled_prompt_embeds = models[2](text_input_ids.to(DEVICE), output_hidden_states=False)[0]  ### text_encoder

            ### Text Encoder 1
            text_input_ids = torch.randn((batch_size, seq_len))
            prompt_embeds = models[1](text_input_ids.to(DEVICE), output_hidden_states=False)    ### text_encoder

            prompt_embeds = prompt_embeds.pooler_output
            prompt_embeds = prompt_embeds.to(dtype=models[1].dtype, device=DEVICE)
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=DEVICE, dtype=dtype)
            ### now have prompt_embeds and pooled_prompt_embeds and text_ids that can feed into the inputs

            
            # Prepare inputs for FluxTransformer
            inputs = {
                "hidden_states": hidden_states, 
                "encoder_hidden_states": prompt_embeds,
                "pooled_projections": pooled_prompt_embeds,
                "timestep": timestep,
                "img_ids": img_ids,
                "txt_ids": text_ids,
            }

            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                with amp_context:
                    output = transformer(**inputs).sample
                    loss = nn.functional.mse_loss(output, target) 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return (flux_train_step, [transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2], [optimizer], flux_inputs)




    elif model_name == "stable_diffusion_mmdit":
        
        model_card = model_cards[model_name]
        model_cls = model_class[model_name]

        with init_mode:
            with torch.device(dev):

                model = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)

                transformer = model.transformer
                vae = model.vae
                text_encoder = model.text_encoder
                text_encoder_2 = model.text_encoder_2
                tokenizer = model.tokenizer
                tokenizer_2 = model.tokenizer_2
                tokenizer_3 = model.tokenizer_3
                scheduler = model.scheduler

            optimizer = create_optimizer(transformer.parameters())

            if ac == AC.FULL:
                ac_classes = model_ac_classes[model_name]
                apply_ac(transformer, ac_classes)


        def hf_train_step(
                model: nn.Module, optim: optim.Optimizer,
            ):

                in_channels = model.in_channels
                joint_attention_dim = model.joint_attention_dim
                pooled_projection_dim = model.pooled_projection_dim
                embed_dim = 4096

                hidden_states, encoder_hidden_states, pooled_projections, timestep, target = generate_sd3_inputs(
                    batch_size, image_size, seq_len, in_channels, joint_attention_dim, pooled_projection_dim, embed_dim, dtype, dev)

                inputs = {
                    "hidden_states": hidden_states,
                    "encoder_hidden_states": encoder_hidden_states,
                    "pooled_projections": pooled_projections,
                    "timestep": timestep,
                }

                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        noise_pred = model(**inputs).sample
                        loss = torch.nn.functional.mse_loss(hidden_states, target.to(dev))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        return (model.transformer, optimizer, hf_train_step)



    else:
         raise ValueError(f"No setup is available for {model_name}. Please choose from {model_names}")

         

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
    b_args.ac_mode = config["ac"].value
    b_args.image_size = config["image_size"]
    b_args.num_denoising_steps = config["num_denoising_steps"]
    return b_args