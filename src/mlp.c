#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "mlp.h"
#include "matrix.h"

struct mlp mlp_init(unsigned int input_size){
    struct mlp mlp = {.layers = linked_list_init()};
    mlp_matrix_init(&mlp.input, input_size, 1);
    mlp_matrix_randomize(&mlp.input);
    mlp.input_size = input_size;
    return mlp;
}

void mlp_free(struct mlp *mlp){
    mlp_matrix_free(&mlp->input);
    struct node *node = NULL;
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        node = linked_list_get_next(&mlp->layers, node);
        /* THE LAYER'S POINTER WILL BE FREED WITH THE LINKED LIST */
        struct layer *current = (struct layer *)node->data;
        mlp_matrix_free(&current->weights);
        mlp_matrix_free(&current->bias);
    }
    linked_list_free(&mlp->layers);
}

void mlp_add_layer(struct mlp *mlp, unsigned int width){
    struct layer *current = malloc(sizeof(struct layer));
    unsigned int previous_size = mlp->input_size;
    if(mlp->layers.length > 0){
        const struct layer *previous = (struct layer *) ((struct node*)mlp->layers.last)->data;
        previous_size = previous->weights.shape[0];
    } 
    mlp_matrix_init(&current->weights, width, previous_size);
    mlp_matrix_init(&current->bias, width, 1);
    mlp_matrix_randomize(&current->weights);
    linked_list_add(&mlp->layers, current);
}

struct mlp_matrix mlp_invoke(struct mlp *mlp){
    struct mlp_matrix matrix = mlp_matrix_copy(&mlp->input);
    struct node *current_node = NULL;
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        current_node = linked_list_get_next(&mlp->layers, current_node);
        struct layer *layer = (struct layer*)current_node->data;
        mlp_matrix_matmult(&layer->weights, &matrix, &matrix);
        mlp_matrix_add(&matrix, &layer->bias, &matrix);
    }
    return matrix;
}

void mlp_print(const struct mlp *mlp){
    struct node *current = linked_list_get_next(&mlp->layers, NULL);
    printf("(%d, %d) -> ", mlp->input_size, 1);
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        struct mlp_matrix matrix = ((struct layer*)current->data)->weights;
        printf("(%d, %d) -> ", matrix.shape[0], matrix.shape[1]);
        current = linked_list_get_next(&mlp->layers, current);
    }
    struct mlp_matrix tail = ((struct layer*)mlp->layers.last->data)->weights;
    printf("(%d, %d) -> ", tail.shape[0], 1);
    printf(" END\n");
}