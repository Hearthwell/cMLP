#ifndef MLP_MLP_H
#define MLP_MLP_H

#include "common.h"
#include "matrix.h"
#include "dataset.h"

struct Layer{
    struct mlp_matrix weights;
    struct mlp_matrix bias;
    struct mlp_matrix output;
};  

struct mlp{
    struct mlp_matrix input;
    unsigned int input_size;
    struct Vector layers;
};

typedef float (*MlpLoss)(const struct mlp_matrix output, const struct mlp_matrix expected); 
struct Optimizer{
    MlpLoss loss;
    float learning_rate;
};

struct mlp mlp_init(unsigned int input_size);
void mlp_free(struct mlp *mlp);

void mlp_add_layer(struct mlp *mlp, unsigned int width);

struct mlp_matrix mlp_invoke(struct mlp *mlp);
/* RETURNS THE LOSS AFTER THE STEP */
float mlp_gradient_descent_step(struct mlp *mlp, struct DatasetItem item, struct Optimizer optimizer);
void mlp_train(struct mlp *mlp, const struct Dataset *training, const struct Dataset *validation, struct Optimizer optimizer, unsigned int epochs);

void mlp_dump(const struct mlp *mlp, const char *path);
struct mlp mlp_load(const char *path);

struct Optimizer mlp_default_optimizer();

void mlp_print(const struct mlp *mlp); 

/* COMMON LOSS FUNCTIONS */
float mlp_loss_cross_entropy(const struct mlp_matrix output, const struct mlp_matrix expected);
float mlp_loss_quadratic(const struct mlp_matrix output, const struct mlp_matrix expected);
float mlp_loss_abs(const struct mlp_matrix output, const struct mlp_matrix expected);

#endif //MLP_MLP_H