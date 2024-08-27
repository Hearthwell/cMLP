#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "mlp.h"
#include "matrix.h"

#define PARTIAL_DERIV_DELTA_H 0.0001
#define DEFAULT_LEARNING_RATE 0.001

struct mlp mlp_init(unsigned int input_size){
    struct mlp mlp = {.layers = vector_init()};
    mlp_matrix_init(&mlp.input, input_size, 1);
    mlp_matrix_randomize(&mlp.input);
    mlp.input_size = input_size;
    return mlp;
}

void mlp_free(struct mlp *mlp){
    mlp_matrix_free(&mlp->input);
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        struct Layer *current = (struct Layer *)mlp->layers.data[i];
        mlp_matrix_free(&current->weights);
        mlp_matrix_free(&current->bias);
        mlp_matrix_free(&current->output);
        free(current);
    }
    vector_free(&mlp->layers);
}

void mlp_add_layer(struct mlp *mlp, unsigned int width){
    struct Layer *current = malloc(sizeof(struct Layer));
    unsigned int previous_size = mlp->input_size;
    if(mlp->layers.length > 0){
        const struct Layer *previous = (struct Layer *)(mlp->layers.data[mlp->layers.length - 1]);
        previous_size = previous->weights.shape[0];
    } 
    mlp_matrix_init(&current->weights, width, previous_size);
    mlp_matrix_randomize(&current->weights);
    mlp_matrix_init(&current->bias, width, 1);
    mlp_matrix_fill(&current->bias, 0);
    /* TO CACHE PREVIOUS LAYER OUTPUTS WHEN TRAINING */
    mlp_matrix_init(&current->output, width, 1);
    vector_add(&mlp->layers, current);
}

static struct mlp_matrix mlp_invoke_ext(struct mlp *mlp, unsigned int start_layer_idx){
    struct mlp_matrix previous_output = mlp->input;
    if(start_layer_idx > 0) previous_output = ((struct Layer *)mlp->layers.data[start_layer_idx - 1])->output;
    for(unsigned int i = start_layer_idx; i < mlp->layers.length; i++){
        struct Layer *layer = (struct Layer *)mlp->layers.data[i];
        mlp_matrix_matmult(&layer->weights, &previous_output, &layer->output);
        mlp_matrix_add(&layer->output, &layer->bias, &layer->output);
        previous_output = layer->output;
    }
    return previous_output;
}

struct mlp_matrix mlp_invoke(struct mlp *mlp){
    /* NO NEED TO CLEAN MATRIX, WILL BE CLEANED WITH THE MLP */
    return mlp_invoke_ext(mlp, 0);
}

/* SINGLE ITERATION OF GRADIENT DESCENT FOR A PARTICULAR DATASET ITEM */
/* SHOULD RUN THIS ON ENTIRE DATASET TO ACCOMPLISH AN EPOCH */
void mlp_gradient_descent_step(struct mlp *mlp, struct DatasetItem item, struct Optimizer optimizer){
    /* COMPUTE BASELINE LOSS */
    memcpy(mlp->input.values, item.input.values, mlp->input.shape[0] * mlp->input.shape[1] * sizeof(float));
    struct mlp_matrix output = mlp_invoke(mlp);
    float base_loss = optimizer.loss(output, item.expected);
    /* COMPUTE THE NETWORK GRADIENT */
    struct Vector gradient = vector_init();
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        struct Layer *layer_gradient = malloc(sizeof(struct Layer));
        const struct Layer *current = (struct Layer *)mlp->layers.data[i];
        mlp_matrix_init(&layer_gradient->weights, current->weights.shape[0], current->weights.shape[1]);
        mlp_matrix_init(&layer_gradient->bias, current->bias.shape[0], current->bias.shape[1]);
        vector_add(&gradient, layer_gradient); 
    }
    /* GO FROM LAST LAYER TO FIRST LAYER TO NOT CHANGE PRECOMPUTED FIRST LAYER OUTPUT */
    /* FROM FIRST INVOCATION AT BEGINNING OF FUNCTION */
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        const unsigned int idx = mlp->layers.length - 1 - i;
        const struct Layer *current = (struct Layer *)mlp->layers.data[idx];
        const unsigned int weight_count = current->weights.shape[0] * current->weights.shape[1];
        struct Layer *gradient_layer = (struct Layer *)gradient.data[idx];
        /* COMPUTE PARTIAL DERIVATIVE FOR EVERY WEIGHT OF CURRENT LAYER */
        for(unsigned int j = 0; j < weight_count; j++){
            const float previous = current->weights.values[j];
            current->weights.values[j] += PARTIAL_DERIV_DELTA_H;
            /* RUN OUTPUT COMPUTATION */
            struct mlp_matrix output = mlp_invoke_ext(mlp, idx);
            /* COMPUTE NEW LOSS AND PARTIAL DERIVATIVE */
            float updated_loss = optimizer.loss(output, item.expected);
            gradient_layer->weights.values[j] = (updated_loss - base_loss) / PARTIAL_DERIV_DELTA_H;
            /* RESET WEIGHT TO ORIGINAL VALUE */
            current->weights.values[j] = previous;
        }
        /* TODO, SAME COMPUTATION AS WITH THE WEIGHTS, SO COMBINE OR MAKE INTO SEPERATE FUNCTION */
        const unsigned int bias_count = current->bias.shape[0] * current->bias.shape[1];
        for(unsigned int j = 0; j < bias_count; j++){
            const float previous = current->bias.values[j];
            current->bias.values[j] += PARTIAL_DERIV_DELTA_H;
            /* RUN OUTPUT COMPUTATION */
            struct mlp_matrix output = mlp_invoke_ext(mlp, idx);
            /* COMPUTE CURRENT LOSS */
            float current_loss = optimizer.loss(output, item.expected);
            gradient_layer->bias.values[j] = (current_loss - base_loss) / PARTIAL_DERIV_DELTA_H;
            current->bias.values[j] = previous;
        }
    }
    /* SCALE AND GO IN OPPOSITE GRADIENT DIRECTION TO PARAMETERS */
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        struct Layer *gradient_layer = (struct Layer *)gradient.data[i];
        mlp_matrix_scale(&gradient_layer->weights, -1.f * optimizer.learning_rate, &gradient_layer->weights);
        mlp_matrix_scale(&gradient_layer->bias, -1.f * optimizer.learning_rate, &gradient_layer->bias);
        struct Layer *layer = (struct Layer *)mlp->layers.data[i];
        mlp_matrix_add(&layer->weights, &gradient_layer->weights, &layer->weights);
        mlp_matrix_add(&layer->bias, &gradient_layer->bias, &layer->bias);
    }
    /* CLEAN GRADIENT */
    for(unsigned int i = 0; i < gradient.length; i++){
        struct Layer *current = (struct Layer *)gradient.data[i];
        mlp_matrix_free(&current->weights);
        mlp_matrix_free(&current->bias);
        free(current);
    }
    vector_free(&gradient);
}

void mlp_train(struct mlp *mlp, const struct Dataset *training, const struct Dataset *validation, struct Optimizer optimizer, unsigned int epochs){
    for(unsigned int k = 0; k < epochs; k++){
        const unsigned int length = training->get_length(training); 
        for(unsigned int i = 0; i < length; i++){
            printf("\rRunning Training, epoch: %d, [%d / %d]", k, i, length);
            fflush(stdout);
            struct DatasetItem item = training->get_element(training, i);
            mlp_gradient_descent_step(mlp, item, optimizer);
            mlp_dataset_element_free(item);
        }
        if(!validation) continue;
        const unsigned int validation_length = validation->get_length(validation);
        float total_loss = 0.f;
        for(unsigned int i = 0; i < validation_length; i++){
            printf("\rRunning Validation, epoch: %d, [%d / %d]", k, i, length);
            fflush(stdout);
            struct DatasetItem item = training->get_element(training, i);
            /* SETUP INPUT */
            for(unsigned int k = 0; k < mlp->input.shape[0] * mlp->input.shape[1]; k++)
                mlp->input.values[k] = item.input.values[k];
            struct mlp_matrix output = mlp_invoke(mlp);
            total_loss += optimizer.loss(output, item.expected);
            mlp_dataset_element_free(item);
        }
        printf("Average Loss For Epoch: %f\n", total_loss / training->get_length(training));
    }
}

struct Optimizer mlp_default_optimizer(){
    return (struct Optimizer) {.loss = mlp_loss_quadratic, .learning_rate = DEFAULT_LEARNING_RATE};
}

void mlp_print(const struct mlp *mlp){
    printf("(%d, %d) -> ", mlp->input_size, 1);
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        struct mlp_matrix matrix = ((struct Layer*)mlp->layers.data[i])->weights;
        printf("(%d, %d) -> ", matrix.shape[0], matrix.shape[1]);
    }
    struct mlp_matrix tail = ((struct Layer*)mlp->layers.data[mlp->layers.length - 1])->weights;
    printf("(%d, %d) -> ", tail.shape[0], 1);
    printf(" END\n");
}

float mlp_loss_cross_entropy(const struct mlp_matrix output, const struct mlp_matrix expected){
    assert("TODO, IMPLEMENT" == NULL);
    (void)output;
    (void)expected;
    return 0.f;
}

float mlp_loss_quadratic(const struct mlp_matrix output, const struct mlp_matrix expected){
    float sum = 0.f;
    for(unsigned int i = 0; i < output.shape[0] * output.shape[1]; i++){
        const float diff = output.values[i] - expected.values[i];
        sum += diff * diff;
    }
    return sum;
}