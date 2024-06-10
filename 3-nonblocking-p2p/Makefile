# 指定编译器
CXX = mpiCC

# 编译器选项
CXXFLAGS = -Wall -g

# 目标文件
TARGET = nonblockingCompareblocking

# 源文件
SRCS = main.cpp smmb.cpp smm.cpp utils.cpp

# 生成的对象文件
OBJS = $(SRCS:.cpp=.o)

# 生成可执行文件
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# 编译源文件生成对象文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理生成的文件
clean:
	rm -f $(OBJS) $(TARGET)