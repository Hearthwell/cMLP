#include <stdio.h>

#include "mlp.h"

#define INPUT_IMG_SIZE 26
#define HIDDEN_DIM 16
#define OUTPUT_DIM 10

int main(){
    printf("DIGIT-RECOGNIZER EXAMPLE\n");
    
    struct mlp network = mlp_init(INPUT_IMG_SIZE * INPUT_IMG_SIZE);
    
    mlp_add_layer(&network, HIDDEN_DIM);
    mlp_add_layer(&network, HIDDEN_DIM);
    mlp_add_layer(&network, OUTPUT_DIM);

    mlp_print(&network); 

    mlp_free(&network);

    return 0;
}