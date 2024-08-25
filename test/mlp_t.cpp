#include <gtest/gtest.h>

extern "C"{
#include "mlp.h"
}

#define MLP_TEST MLP_TEST

TEST(MLP_TEST, mlp_add_layer){
    /* TODO, IMPLEMENT */
}

TEST(MLP_TEST, mlp_invoke){
    struct mlp network = mlp_init(10);
    mlp_add_layer(&network, 20);
    mlp_matrix_fill(&network.input, 1);
    struct mlp_matrix output = mlp_invoke(&network);
    EXPECT_EQ(output.shape[0], 20);
    EXPECT_EQ(output.shape[1], 1);
    mlp_matrix_free(&output);
    mlp_free(&network);
}