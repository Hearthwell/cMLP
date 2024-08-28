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
    dataset.free(&dataset);
}

TEST(DATASET_TEST, dataloader){
    char buffer[MAX_PATH_DIR] = {0};
    char *path = get_ressource_path(buffer, "data/mockDataset");
    struct Dataset dataset = mlp_dataset_mnist_init(path);
    struct Dataset dataloader = mlp_dataloader_init(&dataset, 2);
    EXPECT_EQ(dataloader.get_length(&dataloader), 2);
    struct DatasetItem item = dataloader.get_element(&dataloader, 0);
    EXPECT_EQ(item.input.shape[0], 28 * 28);
    EXPECT_EQ(item.input.shape[1], 2);
    EXPECT_EQ(item.expected.shape[0], 10);
    EXPECT_EQ(item.expected.shape[1], 2);
    /* ADD VERIFICATION TO CONTENT OF MATRICES */
    mlp_dataset_element_free(item);
    dataloader.free(&dataloader);
}