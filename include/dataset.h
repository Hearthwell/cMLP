#ifndef MLP_DATASET_H
#define MLP_DATASET_H

#include "matrix.h"

struct DatasetItem{
    struct mlp_matrix input;
    struct mlp_matrix expected;
};

struct Dataset{
    unsigned int (*get_length)(const struct Dataset *dataset);
    struct DatasetItem (*get_element)(const struct Dataset *dataset, unsigned int idx);
    void (*free)(struct Dataset *dataset);
    void *data;
};

void mlp_dataset_element_free(struct DatasetItem item);

/* DATALOADER ABSTRACTION */
struct Dataset mlp_dataloader_init(struct Dataset *dataset, unsigned int batch_size);
void mlp_dataloader_free(struct Dataset *dataset);

/* MNIST DIGIT-RECOGNITION DATASET IMPLEMENTATION */
struct Dataset mlp_dataset_mnist_init(const char *path);
void mlp_dataset_mnist_free(struct Dataset *dataset);

#endif //MLP_DATASET_H