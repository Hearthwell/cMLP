#include <stdlib.h>

#include "common.h"

struct linked_list linked_list_init(){
    struct linked_list list = {.length = 0, .first = NULL, .last = NULL};
    return list;
}

void linked_list_free(struct linked_list *list){
    struct node *current = list->first;
    for(unsigned int i = 0; i < list->length; i++){
        free(current->data);
        struct node *temp = current;
        free(temp);
        current = current->next;
    }
}

struct node* linked_list_get_next(const struct linked_list *list, struct node *node){
    if(node == NULL) return list->first;
    return node->next;
}

void linked_list_add(struct linked_list *list, void *element){
    struct node *current = malloc(sizeof(struct node));
    current->data = element;
    if(list->last) list->last->next = current;
    list->last = current;
    if(!list->first) list->first = current;
    list->length++;
}