# Multi Layer Perceptron C Library

## If Interested In More Complex Neural Networks Check Out: https://github.com/hearthwell/adci 

## Define Network Architecture And Run Inference And Training 

### Quick Start (Still In Development)
```
# 3Blue1Brown Machine Learning Series digit-recognizer Architecture
struct mlp network = mlp_init(28 * 28);
mlp_add_layer(&network, 16);
mlp_add_layer(&network, 16);
mlp_add_layer(&network, 10);

# WEIGHTS ARE RANDOMIZED WHEN INITIALIZING NETWORK

# TRAINING
struct Dataset dataset = mlp_dataset_mnist_init(TRAINING_DATASET_DIR);
mlp_train(&network, dataset, mlp_default_optimizer(), 1);
mlp_dataset_mnist_free(&dataset);

# NN FORWARD
# TODO, SET NN INPUT BEFORE INFERENCE
struct mlp_matrix output = mlp_invoke(&network);

# CLEANUP
mlp_free(&network);
```

# RoadMap
- [x] Support Network Inference
- [x] Support Network Training
- [ ] Support For Loading/Saving Network From/To File
- [ ] Support Export To Tflite
- [ ] Support Export To Onnx