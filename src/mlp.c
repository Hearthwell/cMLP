#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>

#include "mlp.h"
#include "matrix.h"

#define MLP_HEADER_TOKEN 0xadaddada
#define EPSILON 0.0000001f
#define PARTIAL_DERIV_DELTA_H 0.0001f
#define DEFAULT_LEARNING_RATE 0.001f

/* PRIVATE FUNCTIONS */

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

static struct Vector mlp_compute_neg_gradient(struct mlp *mlp, const struct DatasetItem item, const struct Optimizer optimizer, unsigned int num_threads){
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
        /* SCALE WEIGHTS GRADIENT (LEARNING RATE) */
        mlp_matrix_scale(&gradient_layer->weights, -1.f * optimizer.learning_rate / num_threads, &gradient_layer->weights);

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
        /* SCALE BIAS GRADIENT (LEARNING RATE) */
        mlp_matrix_scale(&gradient_layer->bias, -1.f * optimizer.learning_rate / num_threads, &gradient_layer->bias);
    }
    return gradient;
}

static struct mlp mlp_copy(const struct mlp *mlp){
    struct mlp copy;
    copy.input_size = mlp->input_size;
    copy.input = mlp_matrix_copy(&mlp->input, NULL);
    copy.layers = vector_init();      
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        struct Layer *current = malloc(sizeof(struct Layer));
        current->weights = mlp_matrix_copy(&((struct Layer*)mlp->layers.data[i])->weights, NULL); 
        current->bias = mlp_matrix_copy(&((struct Layer*)mlp->layers.data[i])->bias, NULL);
        current->output = mlp_matrix_copy(&((struct Layer*)mlp->layers.data[i])->output, NULL);
        vector_add(&copy.layers, current);
    }
    return copy;
}

static void mlp_apply_gradient(struct mlp *mlp, struct Vector gradient){
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        struct Layer *gradient_layer = (struct Layer *)gradient.data[i];
        struct Layer *layer = (struct Layer *)mlp->layers.data[i];
        mlp_matrix_add(&layer->weights, &gradient_layer->weights, &layer->weights);
        mlp_matrix_add(&layer->bias, &gradient_layer->bias, &layer->bias);
    }
}

static void mlp_clean_gradient(struct Vector *gradient){
    for(unsigned int i = 0; i < gradient->length; i++){
        struct Layer *current = (struct Layer *)gradient->data[i];
        mlp_matrix_free(&current->weights);
        mlp_matrix_free(&current->bias);
        free(current);
    }
    vector_free(gradient);
}

struct ComputeGradientArgs{
    struct mlp mlp;
    struct DatasetItem item;
    struct Optimizer optimizer;
    unsigned int should_stop;
    unsigned int thread_count;
    sem_t start_sync;
    sem_t done_sync;
    /* THREAD RETURN VALUE */
    struct Vector gradient;
};
static void * mlp_threaded_compute_gradient(void *thread_args){
    struct ComputeGradientArgs *args = (struct ComputeGradientArgs *)thread_args;
    for(;;){
        assert(sem_wait(&args->start_sync) == 0);
        if(args->should_stop) break;
        args->gradient = mlp_compute_neg_gradient(&args->mlp, args->item, args->optimizer, args->thread_count);
        assert(sem_post(&args->done_sync) == 0);
    }
    return NULL;
}
/* END PRIVATE FUNCTIONS */

struct mlp mlp_init(unsigned int input_size){
    struct mlp mlp = {.layers = vector_init()};
    mlp_matrix_init(&mlp.input, input_size, 1);
    mlp_matrix_fill(&mlp.input, 0.f);
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
    mlp_matrix_randomize(&current->bias);
    /* TO CACHE PREVIOUS LAYER OUTPUTS WHEN TRAINING */
    mlp_matrix_init(&current->output, width, 1);
    vector_add(&mlp->layers, current);
}

struct mlp_matrix mlp_invoke(struct mlp *mlp){
    /* NO NEED TO CLEAN MATRIX, WILL BE CLEANED WITH THE MLP */
    return mlp_invoke_ext(mlp, 0);
}

/* SINGLE ITERATION OF GRADIENT DESCENT FOR A PARTICULAR DATASET ITEM */
/* SHOULD RUN THIS ON ENTIRE DATASET TO ACCOMPLISH AN EPOCH */
float mlp_gradient_descent_step(struct mlp *mlp, const struct DatasetItem item, struct Optimizer optimizer){
    struct Vector gradient = mlp_compute_neg_gradient(mlp, item, optimizer, 1);
    /* GO IN OPPOSITE GRADIENT DIRECTION TO PARAMETERS */
    mlp_apply_gradient(mlp, gradient);
    /* CLEAN GRADIENT */
    mlp_clean_gradient(&gradient);
    struct mlp_matrix output = mlp_invoke(mlp);
    return optimizer.loss(output, item.expected);
}

void mlp_train_threaded(struct mlp *mlp, const struct Dataset *training, const struct Dataset *validation, struct Optimizer optimizer, unsigned int epochs, unsigned int num_threads){
    /* INIT THE THREADS */
    pthread_t threads[num_threads];
    struct ComputeGradientArgs args[num_threads];
    for(unsigned int i = 0; i < num_threads; i++){
        args[i].optimizer = optimizer;
        args[i].should_stop = 0;
        pthread_create(&threads[i], NULL, mlp_threaded_compute_gradient, &args[i]);
        sem_init(&args[i].start_sync, 0, 0);
        sem_init(&args[i].done_sync, 0, 0);
    }
    /* RUN ACTUAL TRAINING */
    for(unsigned int k = 0; k < epochs; k++){
        const unsigned int length = (training->get_length(training) / num_threads) + 1; 
        for(unsigned int i = 0; i < length; i++){
            const unsigned int thread_count = (num_threads * i > length) ? length % num_threads : num_threads;
            /* ACTIVATE EACH THREAD */
            for(unsigned int j = 0; j < thread_count; j++){
                /* TODO, MAKE ALGO WORK WITHOUT HAVING TO COPY NN EACH TIME */
                args[j].mlp = mlp_copy(mlp);
                args[j].item = training->get_element(training, num_threads * i + j);
                args[j].thread_count = thread_count;
                assert(sem_post(&args[j].start_sync) == 0);
            }
            /* RETRIEVE GRADIENT AND MODIFY ORIGINAL MODEL PARAMETERS */
            float current_loss = 0.f;
            for(unsigned int j = 0; j < thread_count; j++){
                assert(sem_wait(&args[j].done_sync) == 0);
                mlp_apply_gradient(mlp, args[j].gradient);
                mlp_clean_gradient(&args[j].gradient);
                mlp_free(&args[j].mlp);
                /* OPTIONNAL, NOT REQUIRED TO HAVE LOSS AT EVERY ITERATION */
                mlp_matrix_copy(&args[j].item.input, &mlp->input);
                struct mlp_matrix output = mlp_invoke(mlp);
                current_loss += optimizer.loss(output, args[j].item.expected);
                mlp_dataset_element_free(args[j].item);
            }
            printf("\nTraining, epoch: %d, [%d / %d], Loss: %f", k, i, length, current_loss / thread_count);
            fflush(stdout);
        }
        if(!validation) continue;
        /* FOR NOW VALIDATION WILL BE DONE ON THE MAIN THREAD */
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
        printf("\n===============\n");
        printf("Average Loss For Epoch: %f\n", total_loss / training->get_length(training));
        printf("===============\n");
    }

    /* DESTROY THREADS */
    for(unsigned int i = 0; i < num_threads; i++){
        args[i].should_stop = 1;
        sem_post(&args[i].start_sync);
        pthread_join(threads[i], NULL);
        sem_destroy(&args[i].start_sync);
        sem_destroy(&args[i].done_sync);
    }
}

void mlp_train(struct mlp *mlp, const struct Dataset *training, const struct Dataset *validation, struct Optimizer optimizer, unsigned int epochs){
    mlp_train_threaded(mlp, training, validation, optimizer, epochs, 1);
}

#define WRITE_HELPER(_fd, _dst, _size) assert(write(_fd, _dst, _size) == (int)_size)
void mlp_dump(const struct mlp *mlp, const char *path){
    assert(mlp->layers.length > 0);
    int fd = open(path, O_WRONLY | O_CREAT, 0777);
    assert(fd >= 0);
    WRITE_HELPER(fd, (uint32_t[]){MLP_HEADER_TOKEN}, sizeof(uint32_t));
    WRITE_HELPER(fd, (uint32_t[]){mlp->layers.length}, sizeof(uint32_t));
    for(unsigned int i = 0; i < mlp->layers.length; i++){
        const struct Layer *current = (const struct Layer *)mlp->layers.data[i];
        WRITE_HELPER(fd, (uint32_t*)current->weights.shape, MAX_NUM_DIM * sizeof(uint32_t));
        const unsigned int weight_count = current->weights.shape[0] * current->weights.shape[1];
        WRITE_HELPER(fd, current->weights.values, sizeof(float) * weight_count);
        /* WRITE BIAS */
        WRITE_HELPER(fd, (uint32_t*)current->bias.shape, MAX_NUM_DIM * sizeof(uint32_t));
        const unsigned int bias_count = current->bias.shape[0] * current->bias.shape[1];
        WRITE_HELPER(fd, current->bias.values, sizeof(float) * bias_count);
    }
    close(fd);
}

#define READ_HELPER(_fd, _dst, _size) assert(read(_fd, _dst, _size) == _size)
struct mlp mlp_load(const char *path){
    int fd = open(path, O_RDONLY);
    assert(fd > 0);
    struct mlp network = mlp_init(0);
    uint32_t buffer = 0;
    READ_HELPER(fd, &buffer, sizeof(uint32_t));
    assert(buffer == MLP_HEADER_TOKEN);
    unsigned int num_layers = 0;
    READ_HELPER(fd, &num_layers, sizeof(uint32_t));
    for(unsigned int i = 0; i < num_layers; i++){
        struct Layer *current = malloc(sizeof(struct Layer));
        READ_HELPER(fd, current->weights.shape, MAX_NUM_DIM * sizeof(uint32_t));
        const unsigned int required_size = sizeof(float) * current->weights.shape[0] * current->weights.shape[1];
        current->weights.values = malloc(required_size);
        READ_HELPER(fd, current->weights.values, required_size);
        READ_HELPER(fd, current->bias.shape, MAX_NUM_DIM * sizeof(uint32_t));
        const unsigned int bias_size = sizeof(float) * current->bias.shape[0] * current->bias.shape[1];
        current->bias.values = malloc(bias_size);
        READ_HELPER(fd, current->bias.values, bias_size);
        mlp_matrix_init(&current->output, current->weights.shape[0], 1); 
        vector_add(&network.layers, current);
    }
    close(fd);
    network.input_size = ((struct Layer*)network.layers.data[0])->weights.shape[1];
    mlp_matrix_free(&network.input);
    mlp_matrix_init(&network.input, network.input_size, 1); 
    return network;
}

struct Optimizer mlp_default_optimizer(){
    return (struct Optimizer) {.loss = mlp_loss_abs, .learning_rate = DEFAULT_LEARNING_RATE};
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

/* ONLY MAKES SENSE WHEN THE OUTPUT VALUES ARE ALSO IN [0, 1] RANGE */
float mlp_loss_cross_entropy(const struct mlp_matrix output, const struct mlp_matrix expected){
    float loss = 0.0f;
    /* BATCH DIM */
    for(unsigned int i = 0; i < output.shape[1]; i++){
        for(unsigned int j = 0; j < output.shape[0]; j++){
            const float log_offset = (output.values[i + j * output.shape[1]] > EPSILON) ? logf(output.values[i + j * output.shape[1]]) : 1.f;
            loss += expected.values[i + j * expected.shape[1]] * log_offset;
        }
    }
    return loss;
}

float mlp_loss_abs(const struct mlp_matrix output, const struct mlp_matrix expected){
    float sum = 0.f;
    for(unsigned int i = 0; i < output.shape[0] * output.shape[1]; i++){
        sum += fabs(output.values[i] - expected.values[i]);
    }
    return sum;
}

float mlp_loss_quadratic(const struct mlp_matrix output, const struct mlp_matrix expected){
    float sum = 0.f;
    for(unsigned int i = 0; i < output.shape[0] * output.shape[1]; i++){
        const float diff = output.values[i] - expected.values[i];
        sum += diff * diff;
    }
    return sum;
}