import time
import json
import warnings
from typing import List, Tuple

from utils import (
    LOGGER,
    colorstr,
    export_formats,
    optimize_for_mobile,
    torch,
    try_export,
    file_size
)

from test import DenseNet121

@try_export
def export_torchscript(model, img, file, optimize, prefix=colorstr("TorchScript:")):
    LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
    f = file.with_suffix(".torchscript")
    ts = torch.jit.trace(model, img, strict=False)
    extra_files = {
        "config.txt": json.dumps(
            {"shape": img.shape, "stride": int(max(model.stride)), "names": model.names}
        )
    }
    if optimize:
        optimize_for_mobile(ts)._save_for_lite_interpreter(
            str(f), _extra_files=extra_files
        )
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None




def run(
    file_checkpoint,
    batch_size: int,
    imgsz: Tuple[int],
    device,
    include: List[str],
    half: bool = False,
    optimize: bool = True
):
    t = time.time()
    include = [x.lower() for x in include]
    fmts = tuple(export_formats()["Argument"][1:])
    flags = [x in include for x in fmts]
    assert sum(flags) == len(
        include
    ), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    (jit, onnx, engine) = flags

    model = DenseNet121(classCount=14,isTrained=False)
    model = torch.load(file_checkpoint,map_location=device)

    im = torch.zeros(batch_size, 3, *imgsz).to(device)

    for _ in range(2):
        y = model(im)

    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {
        "stride": int(max(model.stride)),
        "names": model.names,
    }  # model metadata
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file_checkpoint} with output shape {shape} ({file_size(file_checkpoint):.1f} MB)"
    )
    f = [""] * len(fmts)
    warnings.filterwarnings(
        action="ignore", category=torch.jit.TracerWarning
    )
    if jit:  # TorchScript
        f[0], _ = export_torchscript(model, im, file_checkpoint, optimize)

    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        h = "--half" if half else ""  # --half FP16 inference arg
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', file_checkpoint.parent.resolve())}"
            f"\nVisualize:       https://netron.app"
        )
    return f

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(file_checkpoint='/home/eco0936_namnh/hn_lap/export_dl/densenet121.pth',batch_size=1,imgsz=(224,224),device=device,include=['torchscript'],half=False,optimize=False)