#ifndef MLP_MLP_H
#define MLP_MLP_H

#include "common.h"
#include "matrix.h"

struct Layer{
    struct mlp_matrix weights;
    struct mlp_matrix bias;
};  

struct mlp{
    struct mlp_matrix input;
    unsigned int input_size;
    struct LinkedList layers;
};

struct mlp mlp_init(unsigned int input_size);
void mlp_free(struct mlp *mlp);

void mlp_add_layer(struct mlp *mlp, unsigned int width);

struct mlp_matrix mlp_invoke(struct mlp *mlp);

void mlp_print(const struct mlp *mlp); 

#endif //MLP_MLP_H