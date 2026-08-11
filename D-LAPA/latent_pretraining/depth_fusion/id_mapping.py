from pathlib import PurePosixPath


DEPTH_PREFIX_SUITES = {"libero_spatial", "libero_object", "libero_goal"}


def lapa_image_path_to_depth_id(image_path: str) -> str:
    """Convert a LAPA JSONL RGB image path into the Stage-2.5 depth feature id.

    Expected RGB path form:
        images/libero_90/TASK_NAME/demo_0/step_12.jpg

    Expected depth feature id form:
        libero_90_TASK_NAME_demo_0_000012

    Some LIBERO depth-feature shards were generated from depth-image folders and
    insert a literal "depth" component after the suite name:
        libero_spatial_depth_TASK_NAME_demo_0_000012
    """
    if not image_path:
        raise ValueError("image_path is empty")

    normalized = str(image_path).replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    try:
        images_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Could not find 'images' segment in image path: {image_path}") from exc

    try:
        suite = parts[images_index + 1]
        task = parts[images_index + 2]
        demo = parts[images_index + 3]
        step_name = parts[images_index + 4]
    except IndexError as exc:
        raise ValueError(f"Image path is too short to derive a depth id: {image_path}") from exc

    if not demo.startswith("demo_"):
        raise ValueError(f"Expected demo segment like 'demo_0', got {demo!r} in {image_path}")
    if not step_name.startswith("step_"):
        raise ValueError(f"Expected step filename like 'step_0.jpg', got {step_name!r} in {image_path}")

    demo_index = demo.split("_", 1)[1]
    step_stem = PurePosixPath(step_name).stem
    step_index = int(step_stem.split("_", 1)[1])
    suite_prefix = f"{suite}_depth" if suite in DEPTH_PREFIX_SUITES else suite
    return f"{suite_prefix}_{task}_demo_{demo_index}_{step_index:06d}"


def resolve_lapa_sample_id(example: dict, id_key: str = "id", source: str = "auto") -> str:
    if source not in ("auto", "id", "image"):
        raise ValueError(f"Unsupported sample id source: {source}")

    if source in ("auto", "id"):
        sample_id = example.get(id_key)
        if sample_id is not None:
            return str(sample_id)
        if source == "id":
            return None

    image_path = example.get("image")
    if image_path is None:
        return None
    return lapa_image_path_to_depth_id(str(image_path))
