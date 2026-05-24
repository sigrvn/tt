CXX      := clang++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude
LDFLAGS  := -lgtest -lgtest_main -pthread

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/run_tests

TEST_SRCS := $(wildcard tests/*.cpp)
TEST_OBJS := $(TEST_SRCS:tests/%.cpp=$(BUILD_DIR)/%.o)

$(TARGET): $(TEST_OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: tests/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: test clean

test: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
