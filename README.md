## Convert model
- TorchScript
- Onnx
- TensortRT

## Setup enviroments
Use python version 3.9

`
pip install -r requirements.txt
`


## Run
`
python main.py --help
`

```
optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS [WEIGHTS ...]
                        model.pt path(s)
  --batch_size BATCH_SIZE
                        Batch size
  --imgsz IMGSZ [IMGSZ ...]
                        image (h, w)
  --dynamic             ONNX/TensorRT: dynamic axes
  --include INCLUDE [INCLUDE ...]
                        torchscript, onnx, engine
  --train               model.train() mode
  --opset OPSET         ONNX: opset version
  --optimize            TorchScript: optimize for mobile

```


Input:

```python
python main.py --weights densenet121.pt \
               --imgsz 224 224 \
               --include torchscript onnx engine 
```
Output:

```script
PyTorch: starting from densenet121.pt with output shape (1, 14) (27.2 MB)

TorchScript: starting export with torch 1.13.1+cu117...
TorchScript: export success 1.6s, saved as densenet121.torchscript (27.7 MB)

ONNX: starting export with onnx 1.8.1...
ONNX: export success 1.4s, saved as densenet121.onnx (27.0 MB)

ONNX: starting export with onnx 1.8.1...
ONNX: export success 1.4s, saved as densenet121.onnx (27.0 MB)

TensorRT: starting export with TensorRT 8.5.3.1...
[03/02/2023-11:33:45] [TRT] [I] [MemUsageChange] Init CUDA: CPU +291, GPU +0, now: CPU 2635, GPU 1410 (MiB)
[03/02/2023-11:33:46] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +261, GPU +76, now: CPU 2951, GPU 1486 (MiB)
[03/02/2023-11:33:46] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage. See `CUDA_MODULE_LOADING` in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars

[03/02/2023-11:33:46] [TRT] [I] ----------------------------------------------------------------
[03/02/2023-11:33:46] [TRT] [I] Input filename:   densenet121.onnx
[03/02/2023-11:33:46] [TRT] [I] ONNX IR version:  0.0.7
[03/02/2023-11:33:46] [TRT] [I] Opset version:    13
[03/02/2023-11:33:46] [TRT] [I] Producer name:    pytorch
[03/02/2023-11:33:46] [TRT] [I] Producer version: 1.13.1
[03/02/2023-11:33:46] [TRT] [I] Domain:           
[03/02/2023-11:33:46] [TRT] [I] Model version:    0
[03/02/2023-11:33:46] [TRT] [I] Doc string:       
[03/02/2023-11:33:46] [TRT] [I] ----------------------------------------------------------------
[03/02/2023-11:33:46] [TRT] [W] onnx2trt_utils.cpp:377: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
TensorRT: Network Description:
TensorRT:	input "images" with shape (1, 3, 224, 224) and dtype DataType.FLOAT
TensorRT:	output "output" with shape (1, 14) and dtype DataType.FLOAT
TensorRT: building FP 16 engine in densenet121.engine
[03/02/2023-11:33:47] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 2982, GPU 1494 (MiB)
[03/02/2023-11:33:47] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 2982, GPU 1504 (MiB)
[03/02/2023-11:33:47] [TRT] [W] TensorRT was linked against cuDNN 8.6.0 but loaded cuDNN 8.5.0
[03/02/2023-11:33:47] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[03/02/2023-11:38:41] [TRT] [I] Total Activation Memory: 4349937152
[03/02/2023-11:38:41] [TRT] [I] Detected 1 inputs and 1 output network tensors.
[03/02/2023-11:38:41] [TRT] [I] Total Host Persistent Memory: 265712
[03/02/2023-11:38:41] [TRT] [I] Total Device Persistent Memory: 274944
[03/02/2023-11:38:41] [TRT] [I] Total Scratch Memory: 0
[03/02/2023-11:38:41] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 13 MiB, GPU 1160 MiB
[03/02/2023-11:38:41] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 198 steps to complete.
[03/02/2023-11:38:41] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 6.6545ms to assign 3 blocks to 198 nodes requiring 4616192 bytes.
[03/02/2023-11:38:41] [TRT] [I] Total Activation Memory: 4616192
[03/02/2023-11:38:41] [TRT] [W] TensorRT encountered issues when converting weights between types and that could affect accuracy.
[03/02/2023-11:38:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to adjust the magnitude of the weights.
[03/02/2023-11:38:41] [TRT] [W] Check verbose logs for the list of affected weights.
[03/02/2023-11:38:41] [TRT] [W] - 240 weights are affected by this issue: Detected subnormal FP16 values.
[03/02/2023-11:38:41] [TRT] [W] - 201 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.
[03/02/2023-11:38:41] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +13, GPU +15, now: CPU 13, GPU 15 (MiB)
TensorRT: export success 297.5s, saved as densenet121.engine (15.5 MB)

Export complete (303.1s)
Results saved to /home/eco0936_namnh/hn_lap/export_dl
Visualize:       https://netron.app
```
