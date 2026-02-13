# Current issues

The current architecture seems to have trouble compiling to the last layer. While our PyTorch/ONNX model compiles to (20, 1, 1) outputs as expected, the MXQ output target outputs to (57600, 1, 1).

We need to create a more explicit model.