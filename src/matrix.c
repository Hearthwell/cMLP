#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#include "matrix.h"

void mlp_matrix_init(struct mlp_matrix *matrix, unsigned int first, unsigned int second){
    matrix->shape[0] = first;
    matrix->shape[1] = second;
    matrix->values = malloc(first * second * sizeof(float));
    for(unsigned int i = 0; i < first * second; i++)
        matrix->values[i] = 0.f;
}

void mlp_matrix_free(struct mlp_matrix *matrix){
    free(matrix->values);
}

void mlp_matrix_randomize(struct mlp_matrix *matrix){
    for(unsigned int i = 0; i < matrix->shape[0] * matrix->shape[1]; i++)
        matrix->values[i] = ((float)rand() * 2.f / RAND_MAX) - 1.f;
}

void mlp_matrix_fill(struct mlp_matrix *matrix, float value){
    for(unsigned int i = 0; i < matrix->shape[0] * matrix->shape[1]; i++)
        matrix->values[i] = value;
}

void mlp_matrix_matmult(const struct mlp_matrix *left, const struct mlp_matrix *right, struct mlp_matrix *out){
    /* CANNOT HAVE SAME OPTIMIZATION AS WITH ADD, BECAUSE  */
    assert(left->shape[1] == right->shape[0]);
    assert(left->values && right->values);
    const unsigned int output_size = out->shape[0] * out->shape[1];
    void *cleanup = NULL;
    if(out == left || out == right) cleanup = out->values;
    struct mlp_matrix matrix = *out;
    matrix.shape[0] = left->shape[0];
    matrix.shape[1] = right->shape[1];
    const unsigned int required_size = matrix.shape[0] * matrix.shape[1];  
    if(output_size < required_size || cleanup || matrix.values == NULL){
        cleanup = out->values;
        matrix.values = malloc(required_size * sizeof(float));
    }
    /* RUN MATRIX MULTIPLICATION */
    for(unsigned int i = 0; i < matrix.shape[0]; i++){
        for(unsigned int j = 0; j < matrix.shape[1]; j++){
            float current = 0.f;
            for(unsigned int k = 0; k < left->shape[1]; k++)
                current += left->values[i * left->shape[1] + k] * right->values[j + k * right->shape[1]];
            matrix.values[i * matrix.shape[1] + j] = current;
        }
    }
    memcpy(out, &matrix, sizeof(struct mlp_matrix));
    if(cleanup) free(cleanup);
}    

void mlp_matrix_add(const struct mlp_matrix *left, const struct mlp_matrix *right, struct mlp_matrix *out){
    /* CHECK IF BOTH MATRICES ARE BROADCASTABLE */
    /* RIGHT MATRIX CANNOT BE SMALLER THAN LEFT */
    /* ONLY RIGHT MATRIX CAN HAVE ITS VALUES READ MORE THAN ONCE */
    /* SO ONLY MAKE A COPY OF MATRIX IF OUT AND RIGHT ARE EQUAL */
    /* OTHERWISE, SAFE TO JUST REUSE LEFT SINCE EACH VALUE CAN ONLY BE READ ONCE */
    assert(left->shape[1] == right->shape[1] || right->shape[1] == 1);
    assert(left->shape[0] == right->shape[0] || right->shape[0] == 1);
    void *cleanup = NULL;
    if(right == out) cleanup = out->values;
    struct mlp_matrix matrix = *out;
    const unsigned int required_size = left->shape[0] * left->shape[1];
    const unsigned int current_size = matrix.shape[0] * matrix.shape[1]; 
    if(required_size < current_size || cleanup || matrix.values == NULL){
        cleanup = matrix.values;
        matrix.values = malloc(required_size * sizeof(float));
    }
    memcpy(matrix.shape, left->shape, sizeof(unsigned int) * MAX_NUM_DIM);
    for(unsigned int i = 0; i < matrix.shape[0]; i++){
        for(unsigned int j = 0; j < matrix.shape[1]; j++){
            const unsigned int offset = i * matrix.shape[1] + j;
            matrix.values[offset] = left->values[offset] + right->values[(i % right->shape[0]) * right->shape[1] + (j % right->shape[1])];
        }
    }
    memcpy(out, &matrix, sizeof(struct mlp_matrix));
    if(cleanup) free(cleanup);
}

void mlp_matrix_scale(const struct mlp_matrix *matrix, float scale, struct mlp_matrix *out){
    assert(matrix->values);
    const unsigned int required_shape = matrix->shape[0] * matrix->shape[1];
    const unsigned int previous_shape = out->shape[0] * out->shape[1];
    memcpy(out->shape, matrix->shape, sizeof(unsigned int) * MAX_NUM_DIM);
    void *cleanup = NULL;
    if(previous_shape < required_shape){
        cleanup = out->values;
        out->values = malloc(required_shape * sizeof(float));
    }
    for(unsigned int i = 0; i < required_shape; i++)
        out->values[i] = matrix->values[i] * scale;
    if(cleanup) free(cleanup);
}

struct mlp_matrix mlp_matrix_copy(const struct mlp_matrix *matrix, struct mlp_matrix *output){
    if(matrix == output) return *matrix;
    const unsigned int required_size = matrix->shape[0] * matrix->shape[1]; 
    struct mlp_matrix copy = {0};
    if(output) copy = *output;
    const unsigned int current_size = copy.shape[0] * copy.shape[1];
    memcpy(copy.shape, matrix->shape, MAX_NUM_DIM * sizeof(unsigned int));
    if(current_size < required_size){
        if(copy.values) free(copy.values);
        copy.values = malloc(sizeof(float) * required_size);
    }
    for(unsigned int i = 0; i < matrix->shape[0] * matrix->shape[1]; i++)
        copy.values[i] = matrix->values[i];
    if(output) memcpy(output, &copy, sizeof(struct mlp_matrix));
    return copy;
}

void mlp_matrix_print(const struct mlp_matrix *matrix){
    printf("[\n");
    for(unsigned int i = 0; i < matrix->shape[0]; i++){
        printf("\t[");
        for(unsigned int j = 0; j < matrix->shape[1]; j++)
            printf("%f, ", matrix->values[i * matrix->shape[1] + j]);
        printf("],\n");
    }
    printf("]\n");
}