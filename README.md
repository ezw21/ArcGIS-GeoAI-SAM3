# SAM3 for ArcGIS Pro

This repository packages Meta's SAM 3 image segmentation model for use with ArcGIS Pro deep learning tools.

It contains an ArcGIS Python raster function wrapper, an Esri model definition file, and a local copy of the upstream SAM 3 codebase. The wrapper exposes SAM 3 as an ArcGIS Pro model that can be called from `Detect Objects Using Deep Learning` and supports text prompts as a model argument.

## What Was Done

This repo adapts SAM 3 to the ArcGIS Pro deep learning runtime:

- Added an ArcGIS raster function entrypoint in `SAM3.py`.
- Added an Esri model definition in `SAM3.emd`.
- Wired the raster function to the local SAM 3 package under `segment-anything-3/`.
- Added support for `text_prompt` as an ArcGIS model parameter.
- Updated the raster function class name to `SAM3` so ArcGIS can resolve the module and class name consistently.
- Kept a compatibility alias `SAM` in the code so older callers do not break.
- Removed the legacy SAM1 fallback and the unused `segment-anything` dependency.

## Repository Layout

- `SAM3.py`
  ArcGIS Python raster function.
- `SAM3.emd`
  Esri model definition used by ArcGIS Pro.
- `segment-anything-3/`
  Local copy of the upstream SAM 3 implementation.
- `segment-anything-3/sam3/model/sam3.pt`
  Default model weight location expected by the EMD.

## Model Weight Location

The default checkpoint path is defined in `SAM3.emd`:

```json
"ModelFile": "segment-anything-3/sam3/model/sam3.pt"
```

The simplest working layout is:

```text
SAM3/
├── README.md
├── SAM3.py
├── SAM3.emd
└── segment-anything-3/
    └── sam3/
        └── model/
            └── sam3.pt
```

If you keep the checkpoint in that location, ArcGIS Pro can load the model without additional changes.

The raster function also searches a few local fallback locations if needed, including:

- next to the EMD
- under `segment-anything-3/`
- under `segment-anything-3/sam3/model/`
- under `segment-anything-3/checkpoints/`
- under `models/`

If you want to use a different checkpoint location, update `ModelFile` in `SAM3.emd`.

## ArcGIS Pro Parameters

The wrapper exposes these model parameters to ArcGIS Pro:

- `text_prompt`
- `padding`
- `batch_size`
- `box_nms_thresh`
- `points_per_batch`
- `stability_score_thresh`
- `min_mask_region_area`

### Text Prompt Support

`text_prompt` is available as a model argument in ArcGIS Pro.

You can pass a single prompt:

```text
text_prompt 'car'
```

Or multiple comma-separated prompts:

```text
text_prompt 'car, building'
```

The raster function splits the string on commas, runs SAM 3 once per prompt, and writes the prompt text into the output `Class` field.

## Example ArcGIS Pro Usage

```python
with arcpy.EnvManager(
    outputCoordinateSystem='PROJCS["NZGD_2000_New_Zealand_Transverse_Mercator",GEOGCS["GCS_NZGD_2000",DATUM["D_NZGD_2000",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",1600000.0],PARAMETER["False_Northing",10000000.0],PARAMETER["Central_Meridian",173.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]',
    extent="DEFAULT",
    processorType="GPU",
    scratchWorkspace=r""
):
    arcpy.ia.DetectObjectsUsingDeepLearning(
        in_raster="wellington-0075m-urban-aerial-photos-2021.jp2",
        out_detected_objects=r"Z:\ArcGIS Pro\2026_SamLora_Kelp\2026_SamLora_Kelp.gdb\Wellington_Car_T29",
        in_model_definition=r"Z:\DLPK\SAM3\SAM3.emd",
        arguments="text_prompt 'car, building';padding 64;batch_size 4;box_nms_thresh 0.7;points_per_batch 64;stability_score_thresh 0.7;min_mask_region_area 0",
        run_nms="NO_NMS",
        confidence_score_field="Confidence",
        class_value_field="Class",
        max_overlap_ratio=0,
        processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
        use_pixelspace="NO_PIXELSPACE",
        in_objects_of_interest=None,
    )
```

## Notes

- `InferenceFunction` in `SAM3.emd` points to `SAM3.py`.
- ArcGIS resolves the raster function class by module name, so the implementation class is named `SAM3`.
- The output feature class stores prompt text in the `Class` field and the stability score in the `Confidence` field.
- The repo is intended to run inside ArcGIS Pro's Python environment with `arcpy` available.

## Upstream SAM 3

The vendored implementation in `segment-anything-3/` is based on the upstream SAM 3 project from Meta. See the upstream documentation in `segment-anything-3/README.md` for the original model details and training/runtime requirements.
