#include <stdlib.h>

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