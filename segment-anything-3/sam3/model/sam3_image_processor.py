# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from typing import Dict, List

import numpy as np
import PIL
import torch
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2


class _ToDtypeCompat:
    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale
        try:
            self.transform = v2.ToDtype(dtype, scale=scale)
        except TypeError:
            self.transform = None

    def __call__(self, image):
        if self.transform is not None:
            return self.transform(image)

        if not self.scale:
            return image.to(self.dtype)

        if self.dtype == torch.uint8:
            if image.dtype.is_floating_point:
                image = image.clamp(0, 1) * 255.0
            return image.to(torch.uint8)

        if self.dtype == torch.float32:
            was_uint8 = image.dtype == torch.uint8
            image = image.to(torch.float32)
            if was_uint8:
                return image / 255.0
            return image

        return image.to(self.dtype)


def _to_image_compat(image):
    try:
        return v2.functional.to_image(image)
    except AttributeError:
        pass

    if isinstance(image, PIL.Image.Image):
        array = image
        width, height = array.size
        channels = len(array.getbands())
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(array.tobytes()))
        tensor = tensor.view(height, width, channels).permute(2, 0, 1)
        return tensor.contiguous()

    if isinstance(image, np.ndarray):
        dtype_map = {
            np.dtype("uint8"): torch.uint8,
            np.dtype("int16"): torch.int16,
            np.dtype("int32"): torch.int32,
            np.dtype("float32"): torch.float32,
            np.dtype("float64"): torch.float64,
        }
        tensor = torch.tensor(
            image.tolist(), dtype=dtype_map.get(np.dtype(image.dtype), torch.float32)
        )
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        return tensor.contiguous()

    if isinstance(image, torch.Tensor):
        tensor = image
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        return tensor.contiguous()

    raise ValueError("Image must be a PIL image, numpy array, or tensor")


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.debug_log_path = None
        self.transform = v2.Compose(
            [
                _ToDtypeCompat(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                _ToDtypeCompat(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _log(self, message):
        if not self.debug_log_path:
            return
        try:
            with open(self.debug_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(message.rstrip() + "\n")
        except Exception:
            pass

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        self._log(
            f"set_image: input_type={type(image).__name__}, original_size=({height},{width}), device={self.device}"
        )
        image = _to_image_compat(image).to(self.device)
        image = self.transform(image).unsqueeze(0)
        self._log(
            "set_image: transformed_tensor="
            f"shape={tuple(image.shape)}, dtype={image.dtype}, device={image.device}, "
            f"min={float(image.min().item()):.4f}, max={float(image.max().item()):.4f}"
        )

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        self._log(
            f"set_image: backbone_out_keys={sorted(state['backbone_out'].keys())}"
        )
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], PIL.Image.Image), (
            "Images must be a list of PIL images"
        )

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(_to_image_compat(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        self._log(f"set_text_prompt: prompt={prompt!r}")
        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        self._log(f"set_text_prompt: text_output_keys={sorted(text_outputs.keys())}")
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )
        self._log(
            f"_forward_grounding: output_keys={sorted(outputs.keys())}"
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)
        self._log(
            "_forward_grounding: raw="
            f"boxes_shape={tuple(out_bbox.shape)}, logits_shape={tuple(out_logits.shape)}, "
            f"masks_shape={tuple(out_masks.shape)}, probs_shape={tuple(out_probs.shape)}"
        )

        keep = out_probs > self.confidence_threshold
        kept_count = int(keep.sum().item()) if keep.numel() else 0
        self._log(
            f"_forward_grounding: confidence_threshold={self.confidence_threshold}, kept={kept_count}"
        )
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        self._log(
            f"_forward_grounding: final_boxes_shape={tuple(boxes.shape)}, final_masks_shape={tuple(state['masks'].shape)}, final_scores_shape={tuple(out_probs.shape)}"
        )
        return state
