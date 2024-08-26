#include <gtest/gtest.h>

#include "common.hpp"

extern "C"{
#include "dataset.h"
}

#define DATASET_TEST DATASET_TEST

TEST(DATASET_TEST, dataset_init){
    char buffer[MAX_PATH_DIR] = {0};
    char *path = get_ressource_path(buffer, "data/mockDataset");
    struct Dataset dataset = mlp_dataset_mnist_init(path);
    EXPECT_EQ(dataset.get_length(&dataset), 4);
    mlp_dataset_mnist_free(&dataset);
}