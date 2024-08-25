# Multi Layer Perceptron C Library

## If Interested In More Complex Neural Networks Check Out: https://github.com/hearthwell/adci 

## Define Network Architecture And Run Inference And Training (In Development) 

### Quick Start (Still In Development)
```
# 3Blue1Brown Machine Learning Series digit-recognizer Architecture
struct mlp network = mlp_init(28 * 28);
mlp_add_layer(&network, 16);
mlp_add_layer(&network, 16);
mlp_add_layer(&network, 10);

# WEIGHTS ARE RANDOMIZED WHEN INITIALIZING NETWORK

# NN FORWARD
struct mlp_matrix output = mlp_invoke(&network);

mlp_matrix_free(&output);
mlp_free(&network);
```

### Since Training Is Not Supported Yet, Inference Is Still Pretty Useless (Training Comming Up)

# RoadMap
- [x] Support Network Inference
- [ ] Support Network Training
- [ ] Support For Loading/Saving Network From/To File
- [ ] Support Export To Tflite
- [ ] Support Export To Onnx