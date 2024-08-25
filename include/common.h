#ifndef MLP_COMMON_H
#define MLP_COMMON_H

struct node{
    void *data;
    struct node *next;
};

struct linked_list{
    struct node *first;
    struct node *last;
    unsigned int length;
};

struct linked_list linked_list_init();
void linked_list_free(struct linked_list *list);

struct node* linked_list_get_next(const struct linked_list *list, struct node *node);
void linked_list_add(struct linked_list *list, void *element);

#endif //MLP_COMMON_H