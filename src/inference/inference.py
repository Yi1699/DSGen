"""AeroGen inference and fixed-noise instance-attention ablations."""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from src.datasets.dataset_display import MyDataset

MODES = ("baseline", "padding", "cross", "self", "both")


def load_state_dict(path):
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    checkpoint = torch.load(path, map_location="cpu", mmap=True)
    return checkpoint.get("state_dict", checkpoint)


def create_model(config_path):
    return instantiate_from_config(OmegaConf.load(config_path).model)


def comparison(images, boxes, labels, output):
    canvas = Image.new("RGB", (512 * len(images), 542), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate(zip(labels, images)):
        image = image.copy()
        overlay = ImageDraw.Draw(image)
        for index, box in enumerate(boxes):
            points = [(int(x * 8), int(y * 8)) for x, y in box.reshape(4, 2)]
            overlay.line(points + points[:1], fill="#ff5555", width=2)
            overlay.text(points[0], str(index + 1), fill="yellow", stroke_width=1, stroke_fill="black")
        canvas.paste(image, (512 * column, 30))
        draw.text((512 * column + 10, 8), label, fill="black")
    canvas.save(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance_attn", choices=MODES, nargs="+", default=["baseline"])
    parser.add_argument("--input_dir", default="demo/txt")
    parser.add_argument("--output_dir", default="outputs/exp1")
    parser.add_argument("--config", default="configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml")
    parser.add_argument("--ckpt", default="ckpt/aerogen_diorr_last.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--self_resolutions", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--precision", choices=("full", "autocast"), default="full")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/"comparisons").mkdir(exist_ok=True)
    (output/"run.json").write_text(json.dumps(vars(args), indent=2))
    dataset = MyDataset(args.input_dir, 512, 64)
    dataset.files.sort()
    if args.limit is not None:
        dataset.files = dataset.files[:args.limit]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    torch.manual_seed(args.seed)
    model = create_model(args.config).cuda().eval()
    state = load_state_dict(args.ckpt)
    model.load_state_dict(state, strict=True)
    del state
    print("Loaded checkpoint (strict=True)", flush=True)
    sampler = DDIMSampler(model)
    unet = model.model.diffusion_model
    unet.instance_self_resolutions = tuple(args.self_resolutions)
    precision = torch.autocast("cuda") if args.precision == "autocast" else nullcontext()
    rows = []
    for index, batch in enumerate(loader):
        stem = Path(batch["filename"][0]).stem
        seed = args.seed + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        controls = {target: [batch[source].float().cuda().repeat_interleave(args.num_samples, 0)]
                    for target, source in (("bbox_control", "bboxes"),
                                           ("category_control", "category_conditions"),
                                           ("mask_control", "mask_conditions"),
                                           ("mask_vector", "mask_vector"))}
        with precision:
            cond = dict(controls, c_crossattn=[model.get_learned_conditioning(batch["txt"] * args.num_samples)])
            uncond = dict(controls, c_crossattn=[model.get_learned_conditioning([""] * args.num_samples)])
        noise = torch.randn((args.num_samples, 4, 64, 64), device="cuda")
        noise_hash = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
        rng_state = torch.cuda.get_rng_state()
        results = {}
        for mode in args.instance_attn:
            unet.instance_attn = mode
            torch.cuda.set_rng_state(rng_state)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with precision:
                samples, _ = sampler.sample(args.steps, args.num_samples, (4, 64, 64), cond,
                                            verbose=False, eta=args.eta, x_T=noise.clone(),
                                            unconditional_guidance_scale=args.guidance_scale,
                                            unconditional_conditioning=uncond)
                decoded = model.decode_first_stage(samples)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert torch.isfinite(decoded).all(), (stem, mode)
            pixels = ((decoded.permute(0, 2, 3, 1).float() * 127.5 + 127.5)
                      .cpu().numpy().clip(0, 255).astype(np.uint8))
            results[mode] = [Image.fromarray(image) for image in pixels]
            folder = output/mode
            folder.mkdir(exist_ok=True)
            for sample, image in enumerate(results[mode]):
                image.save(folder/f"{stem}_{sample}.png")
                rows.append(dict(layout=stem, mode=mode, sample=sample, seed=seed,
                                 instances=int(batch["mask_vector"].sum()), seconds=round(elapsed, 3),
                                 noise_sha256=noise_hash))
            print(f"{index+1}/{len(dataset)} {stem} {mode}: {elapsed:.2f}s", flush=True)
        boxes = batch["bboxes"][0][batch["mask_vector"][0].bool()].numpy()
        for sample in range(args.num_samples):
            images = [Image.new("RGB", (512, 512), "#303030")]
            images += [results[mode][sample] for mode in args.instance_attn]
            comparison(images, boxes, [stem] + args.instance_attn,
                       output/"comparisons"/f"{stem}_{sample}.jpg")
        with (output/"stats.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {len(rows)} images and comparisons to {output}", flush=True)


if __name__ == "__main__":
    main()
