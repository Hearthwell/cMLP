#include <stdio.h>

#include "mlp.h"
#include "dataset.h"

#define INPUT_IMG_SIZE 28
#define HIDDEN_DIM 16
#define OUTPUT_DIM 10

/* TODO, CHANGE FOR YOUR OWN PATH */
#define TRAINING_DATASET_DIR "~/datasets/MNIST/training"
#define VALIDATION_DATASET_DIR "~/datasets/MNIST/validation"

int main(){
    printf("DIGIT-RECOGNIZER EXAMPLE\n");
    
    struct mlp network = mlp_init(INPUT_IMG_SIZE * INPUT_IMG_SIZE);
    mlp_add_layer(&network, HIDDEN_DIM);
    mlp_add_layer(&network, HIDDEN_DIM);
    mlp_add_layer(&network, OUTPUT_DIM);

    mlp_print(&network); 

    const unsigned int num_epochs = 1;

    struct Dataset dataset = mlp_dataset_mnist_init(TRAINING_DATASET_DIR);
    mlp_train(&network, &dataset, NULL, mlp_default_optimizer(), num_epochs);
    /* STILL UNDER DEVELOPMENT, SINGLE THREADED IS THE RECOMMENDED WAY FOR NOW */
    //mlp_train_threaded(&network, &dataset, NULL, mlp_default_optimizer(), num_epochs, 2);
    dataset.free(&dataset);

    mlp_free(&network);

    return 0;
}