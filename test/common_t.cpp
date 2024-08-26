#include <gtest/gtest.h>
#include "common.hpp"

#define COMMON_TEST COMMON_TEST

TEST(COMMON_TEST, get_ressource_path){
    char buffer[MAX_PATH_DIR] = {0};
    char *path = get_ressource_path(buffer, "data");
    printf("%s\n", path);
    EXPECT_NE(path, nullptr);
}