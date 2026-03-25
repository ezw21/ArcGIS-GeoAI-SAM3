import os
import sys
import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageColor

from SAM import _mask_to_polygon_rings


WORKSPACE_ROOT = os.path.dirname(__file__)
SAM3_REPO_ROOT = os.path.join(WORKSPACE_ROOT, "segment-anything-3")
CHECKPOINT_PATH = os.path.join(SAM3_REPO_ROOT, "sam3", "model", "sam3.pt")
OUTPUT_ROOT = Path(WORKSPACE_ROOT) / "sam3_runs"
IMAGE_PATH = r"Z:\Data\Network_Waitaki_04_IMAGES\04_IMAGES\360\R0017247.jpg"
DEVICE = "cuda"
SHARP_COLOR_PALETTE = [
    "#ff3b30",
    "#0a84ff",
    "#ffd60a",
    "#34c759",
    "#ff2d55",
    "#ff9f0a",
    "#5ac8fa",
    "#bf5af2",
    "#64d2ff",
    "#ff453a",
]
PROMPT_SPECS = [
    {
        "prompt": "building",
        "class_name": "building",
        "color": "#ff3b30",
    },
]


if SAM3_REPO_ROOT not in sys.path:
    sys.path.insert(0, SAM3_REPO_ROOT)


from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def slugify(value):
    chars = []
    for char in value.lower().strip():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "output"


def get_next_run_index():
    if not OUTPUT_ROOT.exists():
        return 0

    pattern = re.compile(r"^T(\d+)_sam3_")
    max_index = -1
    for child in OUTPUT_ROOT.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def build_run_directory(image_path, prompt_specs):
    image_slug = slugify(Path(image_path).stem)
    run_index = get_next_run_index()
    run_dir = OUTPUT_ROOT / f"T{run_index}_sam3_{image_slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_output_paths(run_dir, class_name, prompt):
    prompt_slug = slugify(class_name or prompt)
    overlay_path = run_dir / f"{prompt_slug}_overlay.png"
    mask_path = run_dir / f"{prompt_slug}_mask_union.png"
    return overlay_path, mask_path


def build_combined_overlay_path(run_dir, image_path, prompt_specs):
    image_slug = slugify(Path(image_path).stem)
    class_slug = slugify("_".join(spec.get("class_name", spec["prompt"]) for spec in prompt_specs))
    return run_dir / f"sam3_{image_slug}_{class_slug}_combined_overlay.png"


def build_resolved_output_paths(run_dir, image_path):
    image_slug = slugify(Path(image_path).stem)
    overlay_path = run_dir / f"sam3_{image_slug}_resolved_overlay.png"
    mask_path = run_dir / f"sam3_{image_slug}_resolved_mask.png"
    polygon_path = run_dir / f"sam3_{image_slug}_resolved_polygons.json"
    return overlay_path, mask_path, polygon_path


def load_image(image_path):
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except OSError as exc:
        if "jpeg2k" not in str(exc).lower():
            raise

    import arcpy

    raster = arcpy.Raster(image_path)
    raster_array = arcpy.RasterToNumPyArray(raster)
    raster_array = np.asarray(raster_array)
    if raster_array.ndim == 3:
        raster_array = np.moveaxis(raster_array, 0, -1)
    if raster_array.ndim != 3:
        raise ValueError(f"Expected 3D raster array, got shape={raster_array.shape}")

    band_count = raster_array.shape[2]
    if band_count >= 3:
        raster_array = raster_array[:, :, :3]
    else:
        raster_array = np.repeat(raster_array[:, :, :1], 3, axis=2)

    return Image.fromarray(raster_array.astype(np.uint8, copy=False), mode="RGB")


def build_union_mask(masks):
    if not isinstance(masks, torch.Tensor) or masks.shape[0] == 0:
        return None

    mask_tensor = masks.detach()
    if mask_tensor.ndim == 4:
        mask_tensor = mask_tensor[:, 0, :, :]
    mask_tensor = mask_tensor.to("cpu")
    union_mask = (mask_tensor > 0).any(dim=0).to(torch.uint8).numpy() * 255
    return union_mask


def save_overlay(image, union_mask, overlay_path, mask_path, color):
    if union_mask is None:
        return False

    base_rgba = image.convert("RGBA")
    overlay_array = np.zeros((union_mask.shape[0], union_mask.shape[1], 4), dtype=np.uint8)
    overlay_color = ImageColor.getrgb(color)
    overlay_array[union_mask > 0] = [overlay_color[0], overlay_color[1], overlay_color[2], 110]
    overlay_image = Image.fromarray(overlay_array, mode="RGBA")
    blended = Image.alpha_composite(base_rgba, overlay_image)
    blended.save(overlay_path)
    Image.fromarray(union_mask, mode="L").save(mask_path)
    return True


def apply_overlay(base_rgba, union_mask, color):
    if union_mask is None:
        return base_rgba

    overlay_array = np.zeros((union_mask.shape[0], union_mask.shape[1], 4), dtype=np.uint8)
    overlay_color = ImageColor.getrgb(color)
    overlay_array[union_mask > 0] = [overlay_color[0], overlay_color[1], overlay_color[2], 110]
    overlay_image = Image.fromarray(overlay_array, mode="RGBA")
    return Image.alpha_composite(base_rgba, overlay_image)


def summarize_first_mask(masks):
    if not isinstance(masks, torch.Tensor) or masks.shape[0] == 0:
        return None, None

    first_mask = masks[0]
    if first_mask.ndim == 3:
        first_mask = first_mask[0]
    first_mask = first_mask.detach().to("cpu").to(torch.uint8)
    mask_area = int(first_mask.sum().item())
    ys, xs = torch.where(first_mask > 0)
    if ys.numel() > 0 and xs.numel() > 0:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    else:
        bbox = None
    return mask_area, bbox


def sharp_color_from_text(text):
    index = sum(ord(char) for char in text) % len(SHARP_COLOR_PALETTE)
    return SHARP_COLOR_PALETTE[index]


def get_prompt_confidence(scores):
    if not scores:
        return 0.0
    return max(float(value) for value in scores)


def build_confidence_resolved_layers(image_size, prompt_results):
    width, height = image_size
    winner_confidence = np.full((height, width), -np.inf, dtype=np.float32)
    winner_index = np.full((height, width), -1, dtype=np.int32)

    for prompt_index, result in enumerate(prompt_results):
        union_mask = result.get("union_mask")
        if union_mask is None:
            continue

        confidence = float(result.get("confidence", 0.0))
        candidate_pixels = union_mask > 0
        update_pixels = candidate_pixels & (confidence >= winner_confidence)
        winner_confidence[update_pixels] = confidence
        winner_index[update_pixels] = prompt_index

    return winner_index, winner_confidence


def save_resolved_outputs(image, prompt_results, run_dir, image_path):
    resolved_overlay_path, resolved_mask_path, resolved_polygon_path = build_resolved_output_paths(run_dir, image_path)
    winner_index, winner_confidence = build_confidence_resolved_layers(image.size, prompt_results)

    if not prompt_results:
        return None

    base_rgba = image.convert("RGBA")
    overlay_array = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    resolved_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    features = []

    for prompt_index, result in enumerate(prompt_results):
        class_mask = winner_index == prompt_index
        if not np.any(class_mask):
            continue

        color = result["color"]
        overlay_color = ImageColor.getrgb(color)
        overlay_array[class_mask] = [overlay_color[0], overlay_color[1], overlay_color[2], 110]
        resolved_mask[class_mask] = 255

        rings = _mask_to_polygon_rings(class_mask.astype(np.uint8, copy=False))
        if not rings:
            continue

        features.append(
            {
                "attributes": {
                    "OID": len(features) + 1,
                    "Class": result["class_name"],
                    "Prompt": result["prompt"],
                    "Confidence": float(winner_confidence[class_mask].max()) if np.any(class_mask) else 0.0,
                },
                "geometry": {
                    "rings": rings,
                },
            }
        )

    if not features:
        return None

    overlay_image = Image.fromarray(overlay_array, mode="RGBA")
    resolved_overlay = Image.alpha_composite(base_rgba, overlay_image)
    resolved_overlay.save(resolved_overlay_path)
    Image.fromarray(resolved_mask, mode="L").save(resolved_mask_path)

    polygon_payload = {
        "displayFieldName": "",
        "fieldAliases": {
            "OID": "OID",
            "Class": "Class",
            "Prompt": "Prompt",
            "Confidence": "Confidence",
        },
        "geometryType": "esriGeometryPolygon",
        "fields": [
            {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
            {"name": "Class", "type": "esriFieldTypeString", "alias": "Class"},
            {"name": "Prompt", "type": "esriFieldTypeString", "alias": "Prompt"},
            {"name": "Confidence", "type": "esriFieldTypeDouble", "alias": "Confidence"},
        ],
        "features": features,
    }
    resolved_polygon_path.write_text(json.dumps(polygon_payload), encoding="utf-8")

    return {
        "overlay_path": resolved_overlay_path,
        "mask_path": resolved_mask_path,
        "polygon_path": resolved_polygon_path,
        "feature_count": len(features),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=IMAGE_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--prompts", nargs="*")
    return parser.parse_args()


def resolve_prompt_specs(cli_prompts):
    if not cli_prompts:
        return PROMPT_SPECS
    return [
        {
            "prompt": prompt,
            "class_name": prompt,
            "color": SHARP_COLOR_PALETTE[index % len(SHARP_COLOR_PALETTE)],
        }
        for index, prompt in enumerate(cli_prompts)
    ]


def build_processor(device):
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=CHECKPOINT_PATH,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=0.1)
    return model, processor


def main():
    args = parse_args()
    image_path = args.image
    device = args.device
    prompt_specs = resolve_prompt_specs(args.prompts)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"SAM3 checkpoint not found: {CHECKPOINT_PATH}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current Python environment")

    model, processor = build_processor(device)

    image = load_image(image_path)
    print(f"image_size={image.size}")
    print(f"device={device}")

    run_dir = build_run_directory(image_path, prompt_specs)
    print(f"run_dir={run_dir}")

    combined_overlay_path = build_combined_overlay_path(run_dir, image_path, prompt_specs)
    prompt_results = []
    inference_state = processor.set_image(image)

    prompt_index = 0
    while prompt_index < len(prompt_specs):
        prompt_spec = prompt_specs[prompt_index]
        prompt = prompt_spec["prompt"]
        class_name = prompt_spec.get("class_name", prompt)
        color = prompt_spec.get("color") or sharp_color_from_text(prompt)
        overlay_path, mask_path = build_output_paths(run_dir, class_name, prompt)

        print(f"prompt={prompt}")
        print(f"class_name={class_name}")
        print(f"color={color}")

        try:
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        except torch.cuda.OutOfMemoryError:
            if device != "cuda":
                raise

            print("cuda_oom_fallback=cpu")
            del inference_state
            del processor
            del model
            gc.collect()
            torch.cuda.empty_cache()

            device = "cpu"
            model, processor = build_processor(device)
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        if isinstance(masks, torch.Tensor):
            mask_count = int(masks.shape[0])
        else:
            mask_count = int(len(masks))

        if isinstance(scores, torch.Tensor):
            score_values = [float(value) for value in scores.detach().cpu().tolist()]
        else:
            score_values = [float(value) for value in scores]

        prompt_confidence = get_prompt_confidence(score_values)

        print(f"mask_count={mask_count}")
        print(f"box_count={len(boxes)}")
        print(f"top_scores={score_values[:10]}")
        print(f"prompt_confidence={prompt_confidence}")

        union_mask = build_union_mask(masks)
        if save_overlay(image, union_mask, overlay_path, mask_path, color):
            print(f"overlay_path={overlay_path}")
            print(f"mask_path={mask_path}")

        prompt_results.append(
            {
                "prompt": prompt,
                "class_name": class_name,
                "color": color,
                "union_mask": union_mask,
                "confidence": prompt_confidence,
            }
        )

        mask_area, bbox = summarize_first_mask(masks)
        if mask_area is not None:
            print(f"first_mask_area={mask_area}")
            print(f"first_mask_bbox={bbox}")

        processor.reset_all_prompts(inference_state)
        if device == "cuda":
            torch.cuda.empty_cache()

        prompt_index += 1

    resolved_outputs = save_resolved_outputs(image, prompt_results, run_dir, image_path)
    if resolved_outputs:
        print(f"combined_overlay_path={resolved_outputs['overlay_path']}")
        print(f"resolved_mask_path={resolved_outputs['mask_path']}")
        print(f"resolved_polygon_path={resolved_outputs['polygon_path']}")
        print(f"resolved_feature_count={resolved_outputs['feature_count']}")
    else:
        print(f"combined_overlay_path={combined_overlay_path}")


if __name__ == "__main__":
    main()