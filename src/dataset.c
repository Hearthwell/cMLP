#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <dirent.h>

#include "dataset.h"
#include "common.h"
#include "matrix.h"
#include "stb_image.h"

#define NUM_DIGITS 10

#ifndef DT_REG
#define DT_REG 0
#endif

void mlp_dataset_element_free(struct DatasetItem item){
    mlp_matrix_free(&item.expected);
    mlp_matrix_free(&item.input);
}

struct DataLoaderMetaData{
    struct Dataset *dataset;
    unsigned int batch_size;
};

static unsigned int mlp_dataloader_get_length(const struct Dataset *dataset){
    struct DataLoaderMetaData *dataloader = (struct DataLoaderMetaData *)dataset->data;
    return dataloader->dataset->get_length(dataloader->dataset) / dataloader->batch_size;
}

static struct DatasetItem mlp_dataloader_get_element(const struct Dataset *dataloader, unsigned int idx){
    struct DataLoaderMetaData *dataloader_metadata = (struct DataLoaderMetaData *)dataloader->data;
    assert(mlp_dataloader_get_length(dataloader) > idx);
    const struct Dataset *dataset = dataloader_metadata->dataset;
    struct DatasetItem sample = dataset->get_element(dataset, 0);
    /* EXPECT ALL DATATSET INPUTS TO BE IN THE SHAPE [N, 1] */
    struct DatasetItem item = {0};
    mlp_matrix_init(&item.input, sample.input.shape[0], dataloader_metadata->batch_size);  
    mlp_matrix_init(&item.expected, sample.expected.shape[0], dataloader_metadata->batch_size);
    mlp_dataset_element_free(sample); 
    for(unsigned int i = 0; i < dataloader_metadata->batch_size; i++){
        struct DatasetItem current = dataset->get_element(dataset, i);
        for(unsigned int k = 0; k < item.input.shape[0]; k++)
            item.input.values[i + k * dataloader_metadata->batch_size] = current.input.values[k];
        for(unsigned int k = 0; k < item.expected.shape[0]; k++)
            item.expected.values[i + k * dataloader_metadata->batch_size] = current.expected.values[k];
        mlp_dataset_element_free(current); 
    }
    return item;
}

struct Dataset mlp_dataloader_init(struct Dataset *dataset, unsigned int batch_size){
    struct DataLoaderMetaData *metadata = malloc(sizeof(struct DataLoaderMetaData));
    metadata->dataset = dataset;
    metadata->batch_size = batch_size;
    struct Dataset dataloader = {.get_length = mlp_dataloader_get_length, .get_element = mlp_dataloader_get_element, .free = mlp_dataloader_free, .data = metadata};
    return dataloader;
}

void mlp_dataloader_free(struct Dataset *dataloader){
    struct DataLoaderMetaData *metadata = (struct DataLoaderMetaData *)dataloader->data;
    if(metadata->dataset->free)
        metadata->dataset->free(metadata->dataset);
    free(metadata);
}

/* MNIST DATASET IMPLEMENTATION */
struct MnistEntry{
    char *image_name;
    unsigned int digit;
};
struct MnistMetaData{
    char *path;
    struct Vector entries;
};

static unsigned int mnist_get_length(const struct Dataset *dataset){
    return ((struct MnistMetaData *)dataset->data)->entries.length;
};

static struct DatasetItem mnist_get_element(const struct Dataset *dataset, unsigned int idx){
    const struct MnistMetaData *mnist_metadata = (struct MnistMetaData *)dataset->data;
    const struct MnistEntry *entry = (struct MnistEntry *)(mnist_metadata->entries.data[idx]);
    struct DatasetItem item = {0};
    const unsigned int base_path_len = strlen(mnist_metadata->path);
    const unsigned int img_name_len = strlen(entry->image_name);
    char filename[base_path_len + 3 + img_name_len + 1];
    strcpy(filename, mnist_metadata->path);
    filename[base_path_len] = '/';
    filename[base_path_len + 1] = (char)('0' + entry->digit);
    filename[base_path_len + 2] = '/';
    strcpy(filename + base_path_len + 3, entry->image_name);
    filename[sizeof(filename) - 1] = '\0';
    int x, y, channel;
    unsigned char *data = stbi_load(filename, &x, &y, &channel, 0);
    mlp_matrix_init(&item.input, x * y, 1);
    for(int i = 0; i < x * y; i++)
        item.input.values[i] = (float)data[i] / 255.0f;
    stbi_image_free(data);
    mlp_matrix_init(&item.expected, 10, 1);
    mlp_matrix_fill(&item.expected, 0.f);
    item.expected.values[entry->digit] = 1.f;
    return item;
}

struct Dataset mlp_dataset_mnist_init(const char *path){
    assert(path != NULL);
    struct MnistMetaData *mnist_metadata = malloc(sizeof(struct MnistMetaData));
    mnist_metadata->path = malloc(strlen(path) + 1);
    strcpy(mnist_metadata->path, path);
    mnist_metadata->entries = vector_init();
    const unsigned int path_length = strlen(path);
    char current_path[path_length + 3];
    strcpy(current_path, path);
    for(unsigned int i = 0; i < NUM_DIGITS; i++){
        current_path[path_length] = '/';
        current_path[path_length + 1] = (char)('0' + i);
        current_path[path_length + 2] = '\0';
        DIR *dir = opendir(current_path);
        if(dir == NULL) continue;
        struct dirent *current;
        while((current = readdir(dir)) != NULL){
            if(current->d_type != DT_REG) continue;
            struct MnistEntry *entry = malloc(sizeof(struct MnistEntry));
            entry->digit = i;
            entry->image_name = malloc(strlen(current->d_name) + 1);
            strcpy(entry->image_name, current->d_name);
            vector_add(&mnist_metadata->entries, entry);
        }
        closedir(dir);
    }
    if(mnist_metadata->entries.length == 0)
        printf("EMPTY DATASET, MAYBE CHECK PATH\n");
    return (struct Dataset){.get_length = mnist_get_length, .get_element = mnist_get_element, .free = mlp_dataset_mnist_free, .data = mnist_metadata};
}

void mlp_dataset_mnist_free(struct Dataset *dataset){
    struct MnistMetaData *mnist_metadata = (struct MnistMetaData *) dataset->data;
    for(unsigned int i = 0; i < mnist_metadata->entries.length; i++){
        struct MnistEntry *entry = (struct MnistEntry *)mnist_metadata->entries.data[i];
        free(entry->image_name);
        free(entry);
    }
    vector_free(&mnist_metadata->entries);
    free(mnist_metadata->path);
    free(mnist_metadata);
}