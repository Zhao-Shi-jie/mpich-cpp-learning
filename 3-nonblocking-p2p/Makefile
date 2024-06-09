# 指定编译器
CXX = mpiCC

# 编译器选项
CXXFLAGS = -Wall -g

TARGET = nonblocking_matrix_multiplication

# 源文件
SRCS = main.cpp utils.cpp

OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)