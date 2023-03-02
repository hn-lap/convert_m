import argparse
import json
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from utils import LOGGER, colorstr, export_formats, file_size, try_export


@try_export
def export_torchscript(
    model: torch.ModuleDict,
    img: torch.Tensor,
    file: Path,
    optimize: bool,
    prefix=colorstr("TorchScript:"),
):
    LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
    f = file.with_suffix(".torchscript")
    ts = torch.jit.trace(model, img, strict=False)
    extra_files = {"config.txt": json.dumps({"shape": img.shape})}
    if optimize:
        optimize_for_mobile(ts)._save_for_lite_interpreter(
            str(f), _extra_files=extra_files
        )
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None


@try_export
def export_onnx(
    model: torch.ModuleDict,
    im: torch.Tensor,
    file: Path,
    opset: int,
    train: bool,
    dynamic: bool,
    prefix=colorstr("ONNX:"),
):
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = file.with_suffix(".onnx")

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.TRAINING
        if train
        else torch.onnx.TrainingMode.EVAL,
        do_constant_folding=not train,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"},
        }
        if dynamic
        else None,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    onnx.save(model_onnx, f)

    return f, model_onnx

@try_export
def export_engine(
    model: torch.ModuleDict,
    file:Path, 
    im: torch.Tensor,
    dynamic: bool,
    verbose: bool = False,
    workspace: int = 4,
    prefix=colorstr("TensorRT:"),
):
    assert (im.device.type != "cpu"), "export running on CPU but must be on GPU"
    import tensorrt as trt
    export_onnx(model, im, file, 13, False, dynamic)
    onnx = file.with_suffix(".onnx")
    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    assert onnx.exists(), f"failed to export ONNX file: {onnx}"
    f = file.with_suffix(".engine")  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f"failed to load ONNX file: {onnx}")
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    LOGGER.info(f"{prefix} Network Description:")
    for inp in inputs:
        LOGGER.info(
            f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}'
        )
    for out in outputs:
        LOGGER.info(
            f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}'
        )

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(
                f"{prefix}WARNING: --dynamic model requires maximum --batch-size argument"
            )
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(
                inp.name,
                (1, *im.shape[1:]),
                (max(1, im.shape[0] // 2), *im.shape[1:]),
                im.shape,
            )
        config.add_optimization_profile(profile)

    LOGGER.info(
        f"{prefix} building FP 16 engine in {f}"
    )
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, "wb") as t:
        t.write(engine.serialize())
    return f, None

def run(
    file_checkpoint: Path,
    batch_size: int,
    imgsz: Tuple[int],
    device: torch.device,
    include: List[str],
    optimize: bool = True,
    opset: int = 12,
    train: bool = False,
    dynamic: bool = False,
):
    t = time.time()
    include = [x.lower() for x in include]
    fmts = tuple(export_formats()["Argument"][1:])
    flags = [x in include for x in fmts]
    assert sum(flags) == len(
        include
    ), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    (jit, onnx, engine) = flags

    
    model = torch.load(file_checkpoint,map_location=device)

    im = torch.zeros(batch_size, 3, *imgsz).to(device)

    for _ in range(2):
        y = model(im.float())

    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape

    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file_checkpoint} with output shape {shape} ({file_size(file_checkpoint):.1f} MB)"
    )
    f = [""] * len(fmts)
    warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning)
    if jit:
        f[0], _ = export_torchscript(model, im, file_checkpoint, optimize)
    if onnx:
        f[1], _ = export_onnx(model, im, file_checkpoint, opset, train, dynamic)
    if engine:
        f[2],_  = export_engine(model,file_checkpoint,im,dynamic,verbose = False,workspace=4)

    f = [str(x) for x in f if x]
    if any(f):
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', Path(file_checkpoint).parent.resolve())}"
            f"\nVisualize:       https://netron.app"
        )
    return f

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(
        file_checkpoint=Path(opt.weights),
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        device=device,
        include=opt.include,
        optimize=opt.optimize,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="densenet121.pt",
        help="model.pt path(s)",
    )
    parser.add_argument('--batch_size',type=int,default=1,help='Batch size')
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[224, 224],
        help="image (h, w)",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="ONNX/TensorRT: dynamic axes"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript",'onnx',"engine"],
        help="torchscript, onnx, engine",
    )
    parser.add_argument("--train", action="store_true", help="model.train() mode")
    parser.add_argument("--opset", type=int, default=12, help="ONNX: opset version")
    parser.add_argument(
        "--optimize", action="store_true", help="TorchScript: optimize for mobile"
    )
    opt = parser.parse_args()
    print(opt)
    main(opt)
    