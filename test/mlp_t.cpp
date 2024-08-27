#include <gtest/gtest.h>

#include "common.hpp"

extern "C"{
#include "mlp.h"
#include "dataset.h"
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
    mlp_free(&network);
}

TEST(MLP_TEST, mlp_loss_quadratic){
    struct mlp_matrix input;
    mlp_matrix_init(&input, 2, 1);
    input.values[0] = 0.5F;
    input.values[1] = 1.5F;
    struct mlp_matrix expected;
    mlp_matrix_init(&expected, 2, 1);
    mlp_matrix_fill(&expected, 2.f);
    float loss = mlp_loss_quadratic(input, expected);
    EXPECT_FLOAT_EQ(loss, 2.5f);
    mlp_matrix_free(&input);
    mlp_matrix_free(&expected);
}

TEST(MLP_TEST, gradient_descent_step){
    /* SIMULATE AND GATE NETWORK */
    struct mlp network = mlp_init(2);
    mlp_add_layer(&network, 2);
    mlp_add_layer(&network, 1);
    network.input.values[0] = 1.f;
    network.input.values[1] = 1.f;
    struct mlp_matrix expected;
    mlp_matrix_init(&expected, 1, 1);
    expected.values[0] = 1.0f;

    struct Optimizer optimizer = mlp_default_optimizer();
    float previous_loss = optimizer.loss(mlp_invoke(&network), expected);
    mlp_gradient_descent_step(&network, (struct DatasetItem){.input = network.input, .expected = expected}, optimizer);
    network.input.values[0] = 1.f;
    network.input.values[1] = 1.f;
    float loss = optimizer.loss(mlp_invoke(&network), expected);
    EXPECT_GT(previous_loss, loss);
    mlp_matrix_free(&expected);
    mlp_free(&network);
}

TEST(MLP_TEST, mlp_save_load){
    struct mlp initial = mlp_init(10);
    mlp_add_layer(&initial, 20);
    mlp_add_layer(&initial, 30);
    struct mlp_matrix input;
    mlp_matrix_init(&input, 10, 1); 
    mlp_matrix_randomize(&input);

    for(unsigned int i = 0; i < input.shape[0]; i++)
        initial.input.values[i] = input.values[i];
    struct mlp_matrix initial_output = mlp_invoke(&initial);

    char buffer[256];
    mlp_dump(&initial, get_ressource_path(buffer, "test.cmlp"));
    struct mlp secondary = mlp_load(buffer);
    EXPECT_EQ(initial.input_size, secondary.input_size);
    EXPECT_EQ(initial.layers.length, secondary.layers.length);
    for(unsigned int i = 0; i < input.shape[0]; i++)
        secondary.input.values[i] = input.values[i];
    struct mlp_matrix output = mlp_invoke(&secondary);
    EXPECT_EQ(initial_output.shape[0], output.shape[0]);
    EXPECT_EQ(initial_output.shape[1], output.shape[1]);
    for(unsigned int i = 0; i < output.shape[0] * output.shape[1]; i++)
        EXPECT_FLOAT_EQ(initial_output.values[i], output.values[i]);
    mlp_matrix_free(&input);
    mlp_free(&initial);
    mlp_free(&secondary);
}