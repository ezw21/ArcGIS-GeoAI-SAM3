import json
import sys, os, importlib
from typing import Optional
sys.path.append(os.path.dirname(__file__))

import numpy as np
import math
import arcpy


def _append_debug_log(log_path, message):
    if not log_path:
        return
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message.rstrip() + "\n")
    except Exception:
        pass


def _first_existing_file(paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def _parse_text_prompts(prompt_text):
    if prompt_text is None:
        return []
    if not isinstance(prompt_text, str):
        prompt_text = str(prompt_text)
    prompts = [prompt.strip() for prompt in prompt_text.split(",")]
    return [prompt for prompt in prompts if prompt]


def _resolve_local_sam3_checkpoint(json_info, model, model_as_file, prf_root_dir):
    search_roots = []
    if model_as_file:
        search_roots.append(os.path.dirname(os.path.abspath(model)))
    search_roots.extend(
        [
            prf_root_dir,
            os.path.join(prf_root_dir, "segment-anything-3"),
            os.path.join(prf_root_dir, "segment-anything-3", "sam3", "model"),
            os.path.join(prf_root_dir, "segment-anything-3", "checkpoints"),
            os.path.join(prf_root_dir, "models"),
        ]
    )

    # Prefer explicit checkpoint references from the EMD when present.
    checkpoint_keys = [
        "ModelFile",
        "ModelPath",
        "Checkpoint",
        "CheckpointPath",
        "Weights",
        "WeightFile",
    ]
    candidates = []
    for key in checkpoint_keys:
        value = json_info.get(key)
        if not value:
            continue
        if os.path.isabs(value):
            candidates.append(value)
        else:
            for root in search_roots:
                candidates.append(os.path.join(root, value))

    # Fallback to common local SAM3 checkpoint names.
    for root in search_roots:
        candidates.extend(
            [
                os.path.join(root, "sam3.pt"),
                os.path.join(root, "sam3.pth"),
                os.path.join(root, "checkpoints", "sam3.pt"),
                os.path.join(root, "checkpoints", "sam3.pth"),
            ]
        )

    return _first_existing_file(candidates)


def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available

def get_centroid(polygon):
    polygon = np.array(polygon)
    return [polygon[:, 0].mean(), polygon[:, 1].mean()]        

def check_centroid_in_center(centroid, start_x, start_y, chip_sz, padding):
    return ((centroid[1] >= (start_y + padding)) and                  (centroid[1] <= (start_y + (chip_sz - padding))) and                 (centroid[0] >= (start_x + padding)) and                 (centroid[0] <= (start_x + (chip_sz - padding))))

def find_i_j(centroid, n_rows, n_cols, chip_sz, padding, filter_detections):
    for i in range(n_rows):
        for j in range(n_cols):
            start_x = i * chip_sz
            start_y = j * chip_sz

            if (centroid[1] > (start_y)) and (centroid[1] < (start_y + (chip_sz))) and (centroid[0] > (start_x)) and (centroid[0] < (start_x + (chip_sz))):
                in_center = check_centroid_in_center(centroid, start_x, start_y, chip_sz, padding)
                if filter_detections:
                    if in_center: 
                        return i, j, in_center
                else:
                    return i, j, in_center
    return None 

def calculate_rectangle_size_from_batch_size(batch_size):
    """
    calculate number of rows and cols to composite a rectangle given a batch size
    :param batch_size:
    :return: number of cols and number of rows
    """
    rectangle_height = int(math.sqrt(batch_size) + 0.5)
    rectangle_width = int(batch_size / rectangle_height)

    if rectangle_height * rectangle_width > batch_size:
        if rectangle_height >= rectangle_width:
            rectangle_height = rectangle_height - 1
        else:
            rectangle_width = rectangle_width - 1

    if (rectangle_height + 1) * rectangle_width <= batch_size:
        rectangle_height = rectangle_height + 1
    if (rectangle_width + 1) * rectangle_height <= batch_size:
        rectangle_width = rectangle_width + 1

    # swap col and row to make a horizontal rect
    if rectangle_height > rectangle_width:
        rectangle_height, rectangle_width = rectangle_width, rectangle_height

    if rectangle_height * rectangle_width != batch_size:
        return batch_size, 1

    return rectangle_height, rectangle_width
    
def get_tile_size(model_height, model_width, padding, batch_height, batch_width):
    """
    Calculate request tile size given model and batch dimensions
    :param model_height:
    :param model_width:
    :param padding:
    :param batch_width:
    :param batch_height:
    :return: tile height and tile width
    """
    tile_height = (model_height - 2 * padding) * batch_height
    tile_width = (model_width - 2 * padding) * batch_width

    return tile_height, tile_width
    
def tile_to_batch(
    pixel_block, model_height, model_width, padding, fixed_tile_size=True, **kwargs
):
    pixel_block = np.asarray(pixel_block)
    inner_width = model_width - 2 * padding
    inner_height = model_height - 2 * padding

    band_count, pb_height, pb_width = pixel_block.shape
    pixel_type = np.dtype(pixel_block.dtype)

    if fixed_tile_size is True:
        batch_height = kwargs["batch_height"]
        batch_width = kwargs["batch_width"]
    else:
        batch_height = math.ceil((pb_height - 2 * padding) / inner_height)
        batch_width = math.ceil((pb_width - 2 * padding) / inner_width)

    batch_shape = (batch_width * batch_height, band_count, model_height, model_width)
    batch = np.empty(batch_shape, dtype=pixel_type)
    batch.fill(0)
    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        # pixel block might not be the shape (band_count, model_height, model_width)
        sub_pixel_block = pixel_block[
            :,
            y * inner_height : y * inner_height + model_height,
            x * inner_width : x * inner_width + model_width,
        ]
        sub_pixel_block_shape = sub_pixel_block.shape
        batch[
            b, :, : sub_pixel_block_shape[1], : sub_pixel_block_shape[2]
        ] = sub_pixel_block

    return batch, batch_height, batch_width


def _prepare_binary_mask_for_cv(mask):
    mask_array = np.asarray(mask)
    if mask_array.ndim > 2:
        mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={getattr(mask_array, 'shape', None)}")
    mask_array = np.ascontiguousarray(mask_array.astype(np.uint8, copy=False))
    if mask_array.max(initial=0) > 1:
        mask_array = (mask_array > 0).astype(np.uint8, copy=False)
        mask_array = np.ascontiguousarray(mask_array)
    return mask_array


def _signed_ring_area(ring):
    area = 0.0
    for idx in range(len(ring) - 1):
        x1, y1 = ring[idx]
        x2, y2 = ring[idx + 1]
        area += (x1 * y2) - (x2 * y1)
    return area / 2.0


def _simplify_ring(ring):
    if len(ring) < 4:
        return ring
    simplified = [ring[0]]
    for idx in range(1, len(ring) - 1):
        prev_x, prev_y = simplified[-1]
        curr_x, curr_y = ring[idx]
        next_x, next_y = ring[idx + 1]
        prev_dx = curr_x - prev_x
        prev_dy = curr_y - prev_y
        next_dx = next_x - curr_x
        next_dy = next_y - curr_y
        if prev_dx == 0 and next_dx == 0:
            continue
        if prev_dy == 0 and next_dy == 0:
            continue
        simplified.append([curr_x, curr_y])
    simplified.append(ring[-1])
    return simplified


def _mask_to_polygon_rings(mask_array, x_offset=0, y_offset=0):
    ys, xs = np.nonzero(mask_array)
    if ys.size == 0 or xs.size == 0:
        return []

    edges = set()

    def add_edge(start, end):
        edge = (start, end)
        reverse = (end, start)
        if reverse in edges:
            edges.remove(reverse)
        else:
            edges.add(edge)

    for y, x in zip(ys.tolist(), xs.tolist()):
        add_edge((x, y), (x + 1, y))
        add_edge((x + 1, y), (x + 1, y + 1))
        add_edge((x + 1, y + 1), (x, y + 1))
        add_edge((x, y + 1), (x, y))

    next_edges = {}
    for start, end in edges:
        next_edges.setdefault(start, []).append(end)

    rings = []
    while next_edges:
        start = next(iter(next_edges))
        current = start
        ring = [[start[0] + x_offset, start[1] + y_offset]]

        while True:
            ends = next_edges.get(current)
            if not ends:
                break
            nxt = ends.pop()
            if not ends:
                del next_edges[current]
            ring.append([nxt[0] + x_offset, nxt[1] + y_offset])
            current = nxt
            if current == start:
                break

        if len(ring) >= 4 and ring[0] == ring[-1]:
            ring = _simplify_ring(ring)
            if len(ring) >= 4:
                rings.append(ring)

    rings.sort(key=lambda ring: abs(_signed_ring_area(ring)), reverse=True)
    return rings
 
features = {
    'displayFieldName': '',
    'fieldAliases': {
        'FID': 'FID',
        'Class': 'Class',
        'Confidence': 'Confidence'
    },
    'geometryType': 'esriGeometryPolygon',
    'fields': [
        {
            'name': 'FID',
            'type': 'esriFieldTypeOID',
            'alias': 'FID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        }
    ],
    'features': []
}

fields = {
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        },
        {
            'name': 'Shape',
            'type': 'esriFieldTypeGeometry',
            'alias': 'Shape'
        }
    ]
}

class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4
    
class SAM:
    def __init__(self):
        self.name = "Segment Anything Model"
        self.description = "This python raster function applies computer vision to segment anything"
        
    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return

        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        device_id = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
                device_id = device
            else:
                arcpy.env.processorType = "CPU"
                device_id = "cpu"
                
        
        
        # Build and initialize SAM3 processor and an adapter that mimics
        # the old SamAutomaticMaskGenerator interface while exposing
        # a `set_text_prompt()` method for text-driven prompts.
        prf_root_dir = os.path.dirname(__file__)
        self.debug_log_path = os.path.join(prf_root_dir, "sam3_arcgis_debug.log")
        try:
            with open(self.debug_log_path, "w", encoding="utf-8") as log_file:
                log_file.write("SAM3 ArcGIS debug log\n")
        except Exception:
            self.debug_log_path = None
        if prf_root_dir not in sys.path:
            sys.path.insert(0, prf_root_dir)

        # Probe candidate parent directories that contain the sam3 package.
        # Supported repo layouts:
        #   segment-anything-3/sam3/   (git clone into segment-anything-3)
        #   sam3/sam3/                 (nested layout)
        #   sam3/                      (flat layout, sam3 is itself the package root)
        _sam3_parent_candidates = [
            os.path.join(prf_root_dir, "segment-anything-3"),
            os.path.join(prf_root_dir, "sam3"),
            prf_root_dir,
        ]
        for _candidate in _sam3_parent_candidates:
            if os.path.isdir(os.path.join(_candidate, "sam3")) and _candidate not in sys.path:
                sys.path.insert(0, _candidate)
                break

        # Try normal package imports first; if ArcGIS prevents package imports
        # fall back to loading modules directly from the repo files.
        importlib.invalidate_caches()
        for module_name in list(sys.modules):
            if module_name == "sam3" or module_name.startswith("sam3."):
                del sys.modules[module_name]

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception:
            # Locate model_builder.py by searching the same candidate paths
            _sam3_pkg_candidates = [
                os.path.join(prf_root_dir, "segment-anything-3", "sam3"),
                os.path.join(prf_root_dir, "sam3", "sam3"),
                os.path.join(prf_root_dir, "sam3"),
            ]
            mb_path = None
            sp_path = None
            for _pkg in _sam3_pkg_candidates:
                _mb = os.path.join(_pkg, "model_builder.py")
                _sp = os.path.join(_pkg, "model", "sam3_image_processor.py")
                if os.path.isfile(_mb):
                    mb_path = _mb
                    sp_path = _sp
                    # Ensure parent of this sam3 package is on sys.path so
                    # relative imports inside sam3 resolve correctly.
                    _pkg_parent = os.path.dirname(_pkg)
                    if _pkg_parent not in sys.path:
                        sys.path.insert(0, _pkg_parent)
                    break
            if mb_path is None:
                raise ImportError(
                    "Cannot locate sam3/model_builder.py. "
                    "Ensure the segment-anything-3 repo is present under Z:\\DLPK\\SAM\\"
                )
            spec = importlib.util.spec_from_file_location("sam3_model_builder", mb_path)
            mb_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mb_mod)
            build_sam3_image_model = getattr(mb_mod, "build_sam3_image_model")

            if sp_path is None or not os.path.isfile(sp_path):
                raise ImportError(
                    "Cannot locate sam3/model/sam3_image_processor.py. "
                    "Ensure the segment-anything-3 repo is intact."
                )
            spec2 = importlib.util.spec_from_file_location("sam3_image_processor", sp_path)
            sp_mod = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(sp_mod)
            Sam3Processor = getattr(sp_mod, "Sam3Processor")

        checkpoint_path = _resolve_local_sam3_checkpoint(
            self.json_info, model, model_as_file, prf_root_dir
        )
        _append_debug_log(self.debug_log_path, f"initialize: device_id={device_id}, checkpoint={checkpoint_path}")
        if checkpoint_path is None:
            local_sam1_ckpt = _first_existing_file(
                [
                    os.path.join(prf_root_dir, "segment-anything", "models", "sam_vit_h_4b8939.pth"),
                    os.path.join(prf_root_dir, "models", "sam_vit_h_4b8939.pth"),
                ]
            )
            msg = (
                "No local SAM3 checkpoint was found. "
                "Add a SAM3 checkpoint file (for example `sam3.pt`) next to the EMD, "
                "under `segment-anything-3`, or reference it from the EMD with `ModelFile`."
            )
            if local_sam1_ckpt is not None:
                msg += (
                    " Found local SAM1 weights at `{}', but those are incompatible with SAM3."
                ).format(local_sam1_ckpt)
            raise Exception(msg)

        # map previous device_id to sam3 device string
        device_str = "cpu" if device_id == "cpu" else "cuda"
        sam3_model = build_sam3_image_model(
            device=device_str,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )
        # instantiate processor with a reasonable default confidence threshold
        self.processor = Sam3Processor(sam3_model, device=device_str, confidence_threshold=0.5)
        self.processor.debug_log_path = self.debug_log_path

        # Adapter to provide a generate(image_np) interface similar to the old SamAutomaticMaskGenerator
        class Sam3MaskGeneratorAdapter:
            def __init__(self, processor):
                self.processor = processor
                self._points_per_batch = 64
                self._min_mask_region_area = 0
                self._stability_score_thresh = float(processor.confidence_threshold)
                self._box_nms_thresh = 0.7
                self.current_text_prompt = None

            def set_text_prompt(self, prompt: Optional[str]):
                self.current_text_prompt = prompt

            def generate(self, image_np):
                from PIL import Image
                import numpy as np
                import torch

                _append_debug_log(
                    getattr(self.processor, "debug_log_path", None),
                    f"adapter.generate: image_shape={tuple(image_np.shape)}, prompt={self.current_text_prompt!r}",
                )
                pil = Image.fromarray(image_np.astype("uint8"))
                state = self.processor.set_image(pil, state=None)
                if self.current_text_prompt:
                    state = self.processor.set_text_prompt(self.current_text_prompt, state=state)

                masks = []
                masks_tensor = state.get("masks")
                scores = state.get("scores")
                if masks_tensor is None:
                    return masks

                if not isinstance(masks_tensor, torch.Tensor):
                    masks_tensor = torch.as_tensor(masks_tensor)

                masks_cpu = masks_tensor.detach().to("cpu")
                if masks_cpu.ndim == 4:
                    masks_cpu = masks_cpu[:, 0, :, :]

                for idx in range(masks_cpu.shape[0]):
                    seg_tensor = masks_cpu[idx].to(torch.uint8).contiguous()
                    area = int(seg_tensor.sum().item())
                    seg = np.array(seg_tensor.tolist(), dtype=np.uint8)
                    score_val = scores[idx]
                    try:
                        score = float(score_val.cpu().item())
                    except Exception:
                        score = float(score_val)
                    masks.append({
                        "segmentation": seg,
                        "area": area,
                        "stability_score": score,
                        "class_name": self.current_text_prompt or "Segment",
                    })
                _append_debug_log(
                    getattr(self.processor, "debug_log_path", None),
                    f"adapter.generate: returned_masks={len(masks)}",
                )
                return masks

            @property
            def points_per_batch(self):
                return self._points_per_batch

            @points_per_batch.setter
            def points_per_batch(self, v):
                self._points_per_batch = int(v)

            @property
            def min_mask_region_area(self):
                return self._min_mask_region_area

            @min_mask_region_area.setter
            def min_mask_region_area(self, v):
                self._min_mask_region_area = int(v)

            @property
            def stability_score_thresh(self):
                return self._stability_score_thresh

            @stability_score_thresh.setter
            def stability_score_thresh(self, v):
                self._stability_score_thresh = float(v)
                # propagate to underlying processor
                try:
                    self.processor.set_confidence_threshold(float(v))
                except Exception:
                    pass

            @property
            def box_nms_thresh(self):
                return self._box_nms_thresh

            @box_nms_thresh.setter
            def box_nms_thresh(self, v):
                self._box_nms_thresh = float(v)

        self.mask_generator = Sam3MaskGeneratorAdapter(self.processor)
        

        
        
            
    def getParameterInfo(self):
        required_parameters = [
            {
                "name": "raster",
                "dataType": "raster",
                "required": True,
                "displayName": "Raster",
                "description": "Input Raster",
            },
            {
                "name": "model",
                "dataType": "string",
                "required": True,
                "displayName": "Input Model Definition (EMD) File",
                "description": "Input model definition (EMD) JSON file",
            },
            {
                "name": "device",
                "dataType": "numeric",
                "required": False,
                "displayName": "Device ID",
                "description": "Device ID",
            },
        ]
        # asdasd
        # optional text prompt for SAM3
        required_parameters.append(
            {
                "name": "text_prompt",
                "dataType": "string",
                "required": False,
                "value": "",
                "displayName": "Text Prompt",
                "description": "Optional text prompt for SAM3 (e.g., 'boat', 'tree')",
            }
        )
        required_parameters.extend(
            [
                {
                    "name": "padding",
                    "dataType": "numeric",
                    "value": int(self.json_info["ImageHeight"]) // 4,
                    "required": False,
                    "displayName": "Padding",
                    "description": "Padding",
                },
                {
                    "name": "batch_size",
                    "dataType": "numeric",
                    "required": False,
                    "value": 4,
                    "displayName": "Batch Size",
                    "description": "Batch Size",
                },
                {
                    "name": "box_nms_thresh",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0.7,
                    "displayName": "box_nms_thresh",
                    "description": "The box IoU cutoff used by non-maximal suppression to filter duplicate masks.",
                },
                {
                    "name": "points_per_batch",
                    "dataType": "numeric",
                    "required": False,
                    "value": 64,
                    "displayName": "points_per_batch",
                    "description": "Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.",
                },
                {
                    "name": "stability_score_thresh",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0.95,
                    "displayName": "stability_score_thresh",
                    "description": "A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions",
                },
                {
                    "name": "min_mask_region_area",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0,
                    "displayName": "min_mask_region_area",
                    "description": "If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.",
                },
                
               
            ]
        )
        return required_parameters
        
        
    def getConfiguration(self, **scalars):
        self.tytx = int(scalars.get("tile_size", self.json_info["ImageHeight"]))
        self.batch_size = (
            int(math.sqrt(int(scalars.get("batch_size", 4)))) ** 2
        )
        self.padding = int(
            scalars.get("padding", self.tytx // 4)
        )
        self.min_mask_region_area = int(
            scalars.get("min_mask_region_area", 0)
        )
        self.points_per_batch = int(
            scalars.get("points_per_batch", 64)
        )
        self.stability_score_thresh = float(
            scalars.get("stability_score_thresh", 0.95)
        )
        self.box_nms_thresh  = float(
            scalars.get("box_nms_thresh", 0.7)
        )
        # read optional text prompt
        self.text_prompt = scalars.get("text_prompt", "")
        (
            self.rectangle_height,
            self.rectangle_width,
        ) = calculate_rectangle_size_from_batch_size(self.batch_size)
        ty, tx = get_tile_size(
            self.tytx,
            self.tytx,
            self.padding,
            self.rectangle_height,
            self.rectangle_width,
        )
        
        return {
            "inputMask": True,
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "padding": self.padding,
            "batch_size": self.batch_size,
            "tx": tx,
            "ty": ty,
            "fixedTileSize": 1,
        }
        
        
    def getFields(self):
        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        raster_mask = pixelBlocks["raster_mask"]
        raster_pixels = pixelBlocks["raster_pixels"]
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels
        
        # create batch from pixel blocks
        batch, batch_height, batch_width = tile_to_batch(
            raster_pixels,
            self.tytx,
            self.tytx,
            self.padding,
            fixed_tile_size=True,
            batch_height=self.rectangle_height,
            batch_width=self.rectangle_width,
        )
        
        mask_list = []
        score_list = []
        class_list = []
        max_mask_area_ratio = 0.98 if self.padding > 0 else 1.0
        prompt_labels = _parse_text_prompts(self.text_prompt)
        if not prompt_labels:
            prompt_labels = [None]
        
        # set SAM model parameters to user defined values
        self.mask_generator.points_per_batch = self.points_per_batch
        self.mask_generator.min_mask_region_area = self.min_mask_region_area
        self.mask_generator.stability_score_thresh = self.stability_score_thresh
        self.mask_generator.box_nms_thresh = self.box_nms_thresh
        
        # iterate over batch and get segment from model
        for batch_idx,input_pixels in enumerate(batch):
            side = int(math.sqrt(self.batch_size))
            i, j = batch_idx // side, batch_idx % side
            input_pixels = np.moveaxis(input_pixels,0,-1)
            for prompt_label in prompt_labels:
                if hasattr(self.mask_generator, 'set_text_prompt'):
                    self.mask_generator.set_text_prompt(prompt_label)
                _append_debug_log(
                    self.debug_log_path,
                    f"vectorize: batch_idx={batch_idx}, tile_ij=({i},{j}), input_pixels_shape={tuple(input_pixels.shape)}, text_prompt={prompt_label!r}",
                )

                masks = self.mask_generator.generate(input_pixels)
                sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
                _append_debug_log(
                    self.debug_log_path,
                    f"vectorize: batch_idx={batch_idx}, prompt={prompt_label!r}, sorted_masks={len(sorted_anns)}",
                )

                for mask_value in sorted_anns:
                    # Reject only near-full-tile masks, which are usually padding artifacts.
                    area_ratio = mask_value['area'] / float(self.tytx * self.tytx)
                    if area_ratio < max_mask_area_ratio:
                        try:
                            masked_image = _prepare_binary_mask_for_cv(mask_value['segmentation'])
                        except Exception as exc:
                            _append_debug_log(
                                self.debug_log_path,
                                f"vectorize: mask_prepare_failed type={type(mask_value['segmentation']).__name__}, error={exc}",
                            )
                            continue

                        _append_debug_log(
                            self.debug_log_path,
                            f"vectorize: contour_input type={type(masked_image).__name__}, dtype={masked_image.dtype}, shape={masked_image.shape}, contiguous={bool(masked_image.flags['C_CONTIGUOUS'])}",
                        )
                        rings = _mask_to_polygon_rings(
                            masked_image,
                            x_offset=j * self.tytx,
                            y_offset=i * self.tytx,
                        )
                        if not rings:
                            continue
                        _append_debug_log(
                            self.debug_log_path,
                            f"vectorize: polygon_rings={len(rings)}, first_ring_points={len(rings[0])}, area_ratio={area_ratio:.4f}, class_name={mask_value.get('class_name', 'Segment')}",
                        )
                        mask_list.append(rings)
                        score_list.append(mask_value["stability_score"])
                        class_list.append(mask_value.get("class_name", "Segment"))
                    else:
                        _append_debug_log(
                            self.debug_log_path,
                            f"vectorize: skipped_large_mask area_ratio={area_ratio:.4f}, threshold={max_mask_area_ratio:.4f}",
                        )
                
        n_rows = int(math.sqrt(self.batch_size))
        n_cols = int(math.sqrt(self.batch_size))
        padding = self.padding
        keep_masks = []
        keep_scores = []
        keep_classes = []
        _append_debug_log(
            self.debug_log_path,
            f"vectorize: contour_stage mask_list={len(mask_list)}, score_list={len(score_list)}, class_list={len(class_list)}",
        )
       
        for idx, mask in enumerate(mask_list):
            if mask == []:
                continue
            centroid = get_centroid(mask[0]) 
            tytx = self.tytx
            grid_location = find_i_j(centroid, n_rows, n_cols, tytx, padding, True)
            if grid_location is not None:
                i, j, in_center = grid_location
                for poly_id, polygon in enumerate(mask):
                    polygon = np.array(polygon)
                    polygon[:, 0] = polygon[:, 0] - (2*i + 1)*padding  # Inplace operation
                    polygon[:, 1] = polygon[:, 1] - (2*j + 1)*padding  # Inplace operation            
                    mask[poly_id] = polygon.tolist()
                if in_center:
                    keep_masks.append(mask)
                    keep_scores.append(score_list[idx])
                    keep_classes.append(class_list[idx])
        _append_debug_log(
            self.debug_log_path,
            f"vectorize: keep_masks={len(keep_masks)}, keep_scores={len(keep_scores)}, keep_classes={len(keep_classes)}",
        )
            
        final_masks =  keep_masks
        pred_score = keep_scores 
        pred_class = keep_classes
        features['features'] = []
        
        for mask_idx, final_mask in enumerate(final_masks):
            features['features'].append({
                'attributes': {
                    'OID': mask_idx + 1,
                    'Class': pred_class[mask_idx],
                    'Confidence': pred_score[mask_idx]
                },
                'geometry': {
                    'rings': final_mask
                }
        })
            _append_debug_log(
                self.debug_log_path,
                f"vectorize: output_features={len(features['features'])}",
            )
        return {"output_vectors": json.dumps(features)}