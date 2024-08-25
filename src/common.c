#include <stdlib.h>
#include <assert.h>

#include "common.h"

struct LinkedList linked_list_init(){
    struct LinkedList list = {.length = 0, .first = NULL, .last = NULL};
    return list;
}

void linked_list_free(struct LinkedList *list){
    struct Node *current = list->first;
    for(unsigned int i = 0; i < list->length; i++){
        free(current->data);
        struct Node *temp = current;
        current = current->next;
        free(temp);
    }
}

struct Node* linked_list_get_next(const struct LinkedList *list, struct Node *node){
    if(node == NULL) return list->first;
    return node->next;
}

void linked_list_add(struct LinkedList *list, void *element){
    struct Node *current = malloc(sizeof(struct Node));
    current->data = element;
    if(list->last) list->last->next = current;
    list->last = current;
    if(!list->first) list->first = current;
    list->length++;
}


/* VECTOR IMPLEMENTATION */
#define VECTOR_DEFAULT_SIZE 10 
struct Vector vector_init(){
    return (struct Vector){
        .capacity = VECTOR_DEFAULT_SIZE,
        .length = 0,
        .data = malloc(VECTOR_DEFAULT_SIZE * sizeof(void *))
    };
}

void vector_free(struct Vector *vector){
    free(vector->data);
    vector->length = 0;
    vector->capacity = 0;
}

void vector_add(struct Vector *vector, void *element){
    if(vector->length >= vector->capacity){
        vector->capacity *= 2;
        vector->data = realloc(vector->data, vector->capacity * sizeof(void *));
    }
    vector->data[vector->length] = element;
    vector->length++;
}

void * vector_get(struct Vector *vector, unsigned int idx){
    assert(idx < vector->length);
    return vector->data[idx];
}