import warnings
from collections import OrderedDict, namedtuple

import numpy as np
import tensorrt as trt
import torch

warnings.filterwarnings("ignore")


def test_model_jit(model, imsz):
    net = torch.jit.load(model)
    im = torch.zeros(1, 3, *imsz).to(torch.device("cuda"))
    out = net(im.float())
    return out


def test_model_onnx(model):
    import numpy as np
    import onnxruntime as rt

    session = rt.InferenceSession(
        model,
        sess_options=rt.SessionOptions(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_shape = session.get_inputs()[0].shape
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    im = np.zeros(input_shape)
    im = im.astype(np.float32)
    results = session.run([output_name], {input_name: im})
    return results


def test_model_onnx_v2(model):
    import onnx

    model_onnx = onnx.load(model)
    onnx.checker.check_model(model_onnx)
    print(onnx.helper.printable_graph(model_onnx.graph))


class INFERENCE_TENSORRT:
    def __init__(self, weight, device, dynamic: bool):
        self.device = device
        self.fp16 = False
        self.dynamic = dynamic
        self.bindings = OrderedDict()
        self.Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.logger = trt.Logger(trt.Logger.INFO)
        self.model = self._load_model(weight=weight)
        self.context = self.model.create_execution_context()
        self.binding_addrs, self.batch_size = self.allocate_buffers()

    def _load_model(self, weight):
        with open(weight, "rb") as f, trt.Runtime(self.logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        return model

    def allocate_buffers(self):
        for index in range(self.model.num_bindings):
            name = self.model.get_binding_name(index)
            dtype = trt.nptype(self.model.get_binding_dtype(index))
            if self.model.binding_is_input(index):
                if -1 in tuple(self.model.get_binding_shape(index)):  # dynamic
                    self.dynamic = True
                    self.context.set_binding_shape(
                        index, tuple(self.model.get_profile_shape(0, index)[2])
                    )
                if dtype == np.float16:
                    self.fp16 = True
            shape = tuple(self.context.get_binding_shape(index))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = self.Binding(
                name, dtype, shape, im, int(im.data_ptr())
            )

        binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        batch_size = self.bindings["images"].shape[0]

        return binding_addrs, batch_size

    def infer(self, im):
        if self.dynamic and im.shape != self.bindings["images"].shape:
            i_in, i_out = (
                self.model.get_binding_index(x) for x in ("images", "output")
            )
            self.context.set_binding_shape(i_in, im.shape)  # reshape if dynamic
            self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
            self.bindings["output"].data.resize_(
                tuple(self.context.get_binding_shape(i_out))
            )
        s = self.bindings["images"].shape
        assert (
            im.shape == s
        ), f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings["output"].data

        return y


if __name__ == "__main__":
    trt = INFERENCE_TENSORRT(
        weight="./densenet121.engine", device=torch.device("cuda"), dynamic=False
    )

    data = torch.from_numpy(np.zeros((1, 3, 224, 224), dtype=np.float32)).to(
        torch.device("cuda")
    )
    y = trt.infer(im=data)
    print(y)
