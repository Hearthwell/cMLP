#ifndef MLP_MATRIX_H
#define MLP_MATRIX_H

#define MAX_NUM_DIM 2

struct mlp_matrix{
    unsigned int shape[MAX_NUM_DIM];
    float *values;
};

void mlp_matrix_init(struct mlp_matrix *matrix, unsigned int first, unsigned int second);
void mlp_matrix_free(struct mlp_matrix *matrix);

void mlp_matrix_randomize(struct mlp_matrix *matrix);
void mlp_matrix_fill(struct mlp_matrix *matrix, float value);

void mlp_matrix_matmult(const struct mlp_matrix *left, const struct mlp_matrix *right, struct mlp_matrix *out);
void mlp_matrix_add(const struct mlp_matrix *left, const struct mlp_matrix *right, struct mlp_matrix *out);
struct mlp_matrix mlp_matrix_copy(const struct mlp_matrix *matrix);

void mlp_matrix_print(const struct mlp_matrix *matrix);

#endif //MLP_MATRIX_H