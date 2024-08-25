LIB_SRC_DIR:=src
LIB_INC_DIR:=include
OUT_DIR:=out
TEST_DIR:=test

LIB_SRC_FILES:=$(wildcard $(LIB_SRC_DIR)/*.c)
LIB_OBJ_FILES:=$(LIB_SRC_FILES:$(LIB_SRC_DIR)/%=$(OUT_DIR)/%.o)

TEST_SRC_FILES:=$(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ_FILES:=$(TEST_SRC_FILES:$(TEST_DIR)/%=$(OUT_DIR)/%.o)

C_DBG_FLAGS:=-g
CFLAGS:=-Wall -Wextra -Werror -I$(LIB_INC_DIR)
# COMMENT TO REMOVE DEBUG INFO
CFLAGS+=$(C_DBG_FLAGS)

$(OUT_DIR)/%.o:$(LIB_SRC_DIR)/%
	gcc $(CFLAGS) -c $< -o $@ 

$(OUT_DIR)/%.o:$(TEST_DIR)/%
	g++ $(CFLAGS) -c $< -o $@ 

examples/digit-recognizer/main: examples/digit-recognizer/main.c $(LIB_OBJ_FILES)
	gcc $(CFLAGS) $^ -o $@ 

tests: $(LIB_OBJ_FILES) $(TEST_OBJ_FILES)
	g++ $(CFLAGS) $^ -o $@ -lgtest

clean:
	rm -rf $(OUT_DIR)/*
	rm -rf tests
	rm -rf examples/digit-recognizer/main