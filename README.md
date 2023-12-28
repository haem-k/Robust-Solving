# Robust-Solving
Python implementation of [Robust Solving of Optical Motion Capture Data by Denoising [Holden et al. 2018]](https://dl.acm.org/doi/10.1145/3197517.3201302) <br />
Implemented in CUDA 10.0, PyTorch 1.2.0 <br />
Rendering code implemented by @ltepenguin

## Training
Before training, precompute values that are frequently used
This process saves rigid body, local reference frame, and statistics in advance

```{sh}
python3 statistics.py --stats_name [file name]
```

For training the network, provide the name of the model
```{sh}
python3 train.py --load_model_name [model name]
```

## Testing
To test the trained model, provide the name of the model to infer.
```{sh}
python3 infer.py --load_model_name [model name]
```

## Exporting the model into ONNX
Export trained model into ONNX format and check if it is exported properly with ONNX Runtime
```{sh}
python3 export_onnx.py --load_model_name [model name]
```

To run the model in Unity, export test input data
```{sh}
python3 export_test_input.py --load_model_name [model name]
```
<br />

> [!CAUTION]
> This code was implemented in 2019. May be outdated.
