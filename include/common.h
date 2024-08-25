#ifndef MLP_COMMON_H
#define MLP_COMMON_H

struct Node{
    void *data;
    struct Node *next;
};

struct LinkedList{
    struct Node *first;
    struct Node *last;
    unsigned int length;
};

struct LinkedList linked_list_init();
void linked_list_free(struct LinkedList *list);

struct Node* linked_list_get_next(const struct LinkedList *list, struct Node *node);
void linked_list_add(struct LinkedList *list, void *element);

struct Vector{
    unsigned int length;
    unsigned int capacity;
    void **data;
};

struct Vector vector_init();
void vector_free(struct Vector *vector);

void vector_add(struct Vector *vector, void *element);
void * vector_get(struct Vector *vector, unsigned int idx);

#endif //MLP_COMMON_H