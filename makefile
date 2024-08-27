LIB_SRC_DIR:=src
LIB_INC_DIR:=include
OUT_DIR:=out
TEST_DIR:=test

LIB_SRC_FILES:=$(wildcard $(LIB_SRC_DIR)/*.c)
LIB_OBJ_FILES:=$(LIB_SRC_FILES:$(LIB_SRC_DIR)/%=$(OUT_DIR)/%.o)

EXTERNAL_SRC_FILES:=$(wildcard $(LIB_SRC_DIR)/external/*.c)
# ADD FOLDER "EXTERNAL" IN OUT TO AVOID NAME COLLISION
EXTERNAL_OBJ_FILES:=$(EXTERNAL_SRC_FILES:$(LIB_SRC_DIR)/external/%.c=$(OUT_DIR)/%.o)

TEST_SRC_FILES:=$(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ_FILES:=$(TEST_SRC_FILES:$(TEST_DIR)/%=$(OUT_DIR)/%.o)

C_BASIC_FLAGS:=-I$(LIB_INC_DIR)
#C_DBG_FLAGS:=-g
CFLAGS:=-Wall -Wextra $(C_BASIC_FLAGS) -O3
# COMMENT TO REMOVE DEBUG INFO
CFLAGS+=$(C_DBG_FLAGS)
C_LINK_FLAGS:=-lm

$(OUT_DIR)/%.o:$(LIB_SRC_DIR)/%
	gcc $(CFLAGS) -c $< -o $@ 

$(OUT_DIR)/%.o:$(LIB_SRC_DIR)/external/%.c
	gcc $(C_BASIC_FLAGS) -c $< -o $@ 

$(OUT_DIR)/%.o:$(TEST_DIR)/%
	g++ $(CFLAGS) -c $< -o $@ 

examples/digit-recognizer/main: examples/digit-recognizer/main.c $(LIB_OBJ_FILES) $(EXTERNAL_OBJ_FILES)
	gcc $(CFLAGS) $^ -o $@ $(C_LINK_FLAGS)

tests: $(LIB_OBJ_FILES) $(EXTERNAL_OBJ_FILES) $(TEST_OBJ_FILES)
	g++ $(CFLAGS) $^ -o $@ -lgtest $(C_LINK_FLAGS)

clean:
	rm -rf $(OUT_DIR)/*
	rm -rf tests
	rm -rf examples/digit-recognizer/main