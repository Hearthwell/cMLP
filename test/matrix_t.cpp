#include <gtest/gtest.h>

extern "C"{
#include "matrix.h"
}

#define MATRIX_TEST_SUITE MATRIX_TEST_SUITE

TEST(MATRIX_TEST_SUITE, mlp_matrix_matmult){
    struct mlp_matrix first;
    mlp_matrix_init(&first, 4, 4);
    mlp_matrix_fill(&first, 1.f);
    struct mlp_matrix second;
    mlp_matrix_init(&second, 4, 1);
    second.values[0] = 1.f;
    second.values[1] = 2.f;
    second.values[2] = 3.f;
    second.values[3] = 4.f;
    struct mlp_matrix output;
    memset(&output, 0, sizeof(struct mlp_matrix));
    mlp_matrix_matmult(&first, &second, &output);
    EXPECT_EQ(output.shape[0], first.shape[0]);
    EXPECT_EQ(output.shape[1], second.shape[1]);
    EXPECT_FLOAT_EQ(output.values[0], 10.f);
    EXPECT_FLOAT_EQ(output.values[1], 10.f);
    EXPECT_FLOAT_EQ(output.values[2], 10.f);
    EXPECT_FLOAT_EQ(output.values[3], 10.f);
    mlp_matrix_free(&first);
    mlp_matrix_free(&second);
    mlp_matrix_free(&output);
}

TEST(MATRIX_TEST_SUITE, mlp_matrix_add){
    struct mlp_matrix first;
    mlp_matrix_init(&first, 4, 4);
    const unsigned int volume = first.shape[0] * first.shape[1];
    for(unsigned int i = 0; i < volume; i++) first.values[i] = (float)i;
    struct mlp_matrix second;
    mlp_matrix_init(&second, 1, 4);
    mlp_matrix_fill(&second, 1.0f);
    struct mlp_matrix output;
    memset(&output, 0, sizeof(struct mlp_matrix));
    mlp_matrix_add(&first, &second, &output);
    EXPECT_EQ(output.shape[0], first.shape[0]);
    EXPECT_EQ(output.shape[1], first.shape[1]);
    for(unsigned int i = 0; i < volume; i++)
        EXPECT_FLOAT_EQ(output.values[i], static_cast<float>(i + 1));
    mlp_matrix_free(&first);
    mlp_matrix_free(&second);
    mlp_matrix_free(&output);
}

TEST(MATRIX_TEST_SUITE, mlp_matrix_add_dim0){
    struct mlp_matrix first;
    mlp_matrix_init(&first, 4, 4);
    const unsigned int volume = first.shape[0] * first.shape[1];
    for(unsigned int i = 0; i < volume; i++) first.values[i] = (float)i;
    struct mlp_matrix second;
    mlp_matrix_init(&second, 4, 1);
    for(unsigned int i = 0; i < second.shape[0]; i++) second.values[i] = (float)i;
    struct mlp_matrix output;
    memset(&output, 0, sizeof(struct mlp_matrix));
    mlp_matrix_add(&first, &second, &output);
    EXPECT_EQ(output.shape[0], first.shape[0]);
    EXPECT_EQ(output.shape[1], first.shape[1]);
    for(unsigned int i = 0; i < first.shape[1]; i++){
        for(unsigned int j = 0; j < first.shape[0]; j++){
            const unsigned int offset = i + j * first.shape[1];
            EXPECT_FLOAT_EQ(output.values[offset], static_cast<float>(first.values[offset] + j));
        }
    }
    mlp_matrix_free(&first);
    mlp_matrix_free(&second);
    mlp_matrix_free(&output);
}