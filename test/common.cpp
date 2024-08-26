#include <string.h>
#include <stdio.h>

#include "common.hpp"

char * get_ressource_path(char buffer[MAX_PATH_DIR], const char *relative_path){
    int length = snprintf(buffer, MAX_PATH_DIR, "%s", __FILE__);    
    if(length == MAX_PATH_DIR) return NULL;
    for(;length > 0; length--){
        if(buffer[length] == '/') break;
    }
    if(strlen(relative_path) + length >= MAX_PATH_DIR) return NULL;
    strcpy(buffer + length + 1, relative_path);
    return buffer;
}