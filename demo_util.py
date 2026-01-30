# Modified from:
#   TiTok: https://github.com/bytedance/1d-tokenizer/blob/main/demo_util.py

import torch

from omegaconf import OmegaConf
from modeling.restok import ResTok
from autoregressive.models.gpt_vanilla import GPT_models as GPT_models_vanilla
from autoregressive.models.gpt_har import GPT_models as GPT_models_har


def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf


def get_restok_tokenizer(config):
    tokenizer = ResTok(config)
    if config.experiment.get("tokenizer_checkpoint", None):
        state_dict = torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu")
        print(tokenizer.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k):v for k,v in state_dict.items()}, strict=False))
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer


def get_llamagen_generator(config):
    if config.model.generator.get("har", False):
        model_cls = GPT_models_har[config.model.generator.model_type]
    else:
        model_cls = GPT_models_vanilla[config.model.generator.model_type]
    generator = model_cls(config)
    if config.experiment.get("generator_checkpoint", None):
        state_dict = torch.load(config.experiment.generator_checkpoint, map_location="cpu")
        generator.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k):v for k,v in state_dict.items()})
    generator.eval()
    generator.requires_grad_(False)
    return generator


@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              guidance_scale=3.0,
              guidance_decay="constant",
              guidance_scale_pow=3.0,
              randomize_temperature=2.0,
              softmax_temperature_annealing=False,
              num_sample_steps=8,
              device="cuda",
              return_tensor=False,
              **sampling_kwargs):
    generator.eval()
    tokenizer.eval()
    if labels is None:
        # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        num_sample_steps=num_sample_steps,
        **sampling_kwargs)

    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image
