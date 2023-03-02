import torch
import torch.nn as nn
import torchvision
import warnings
warnings.filterwarnings("ignore")

class DenseNet121(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def test_model_jit(model,imsz):
    net = torch.jit.load(model)
    im = torch.zeros(1, 3, *imsz).to(torch.device('cuda'))
    out = net(im.float())
    return out


def test_model_onnx(model):
    import onnxruntime as rt
    import numpy as np  
    session = rt.InferenceSession(
            model, sess_options=rt.SessionOptions(), providers=["CUDAExecutionProvider",'CPUExecutionProvider']
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

def test_model_tensorrt(model):
    pass 

if __name__ == "__main__":
    test_model_onnx_v2('./densenet121.onnx')