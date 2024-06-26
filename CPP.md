## Makefile

```makefile
#编写makefile文件

#方式1
testfiles: main.cpp function.cpp
	g++ -o testfiles main.cpp function.cpp

#方式2，在文件中定义宏和变量
CXX = g++
TARGET= testfiles
OBJ = main.o function.o
$(TARGET):$(OBJ)
	$(CXX) -o $(TARGET) $(OBJ)

main.o:main.cpp
	$(CXX) -c main.cpp
function.o:function.cpp
	$(CXX) -c function.cpp

#方式3，对方式2进行修改
#$@: the target file
#$^: all the prerequisites
#$<: the first prerequisites

CXX = g++
TARGET= testfiles
OBJ = main.o function.o
$(TARGET):$(OBJ)
	$(CXX) -o $@ $^

%.o:%.cpp
	$(CXX) -c $^

#方式4，利用PHONY自动清除已经产生的文件，比如.o

CXX = g++
TARGET= testfiles.exe
OBJ = main.o function.o
$(TARGET):$(OBJ)
	$(CXX) -o $@ $^


%.o:%.cpp
	$(CXX) -c $<

.PHONY:clean
clean:
	del *.o $(TARGET)

#方式5，利用自带函数获得文件
CXX = g++
TARGET = testfiles.exe
CFLAGES = -c -Wall  #-Wall展示编译的warning
SRC = $(wildcard ./*.cpp)#wildcard-找到当前路径下所有的.cpp文件
OBJ = $(patsubst %.cpp,%.o,$(SRC))#patsubst-将.cpp文件替换为.o文件
$(TARGET):$(OBJ)
	$(CXX) -o $(TARGET) $(OBJ)
%.o:%.cpp
	$(CXX) $(CFLAGES) $<

.PHONY:clean
clean:
	del *.o $(TARGET) 

#其他函数
#1.过滤出不以.c结尾的字符串
object=foo.o bar.o baz.c
flitered_objects=$(filter-out %.c $object)
```



## Cmake

问题：Windows下CMake不能生成makefile的问题

解决方案：可能是由于安装了Visual Studio，也可能是windows10默认，CMake会生成`MSVC`解决方案，在构建目录中检查有 .sln 文件。

指定解决方案是Unix 平台的Makefiles

`cmake .. -G "Unix Makefiles"` （第一次运行cmake时）

```cmake
#方式1
cmake_minimum_required(VERSION 3.10)  #确定cmake最低需求的版本

project(Hello)   #构建的可执行文件的名称

##当有大量源文件时，可以利用aux_source_directory来获得指定目录下所有的源文件，并存到一个DIR_SRCS变量中
##aux_source_directory (<dir> <variable>)
aux_source_directory(./src DIR_SRCS)

#当头文件和源文件不在同一个文件夹下，需要指出.h的位置
include_directories(./include)

#第一个参数是可执行文件的名称，第二个参数是源文件
add_executable(Hello ${DIR_SRCS})  
```



## C++使用静态库

在Linux下使用C++静态库的步骤如下：

编写代码并编译成目标文件（.o），比如**`g++ -c test.cpp -o test.o`**。

然后使用**ar命令**将编译生成的.o文件打包成静态库文件（.a），比如 **`ar rcs libtest.a test.o`**。

ar是一个用于创建、修改和提取归档文件的命令行工具。归档文件是一种将多个文件组合成单个文件的方式，通常用于将多个目标文件组合为一个库文件（如静态库）或者打包多个文件以备份或分发。
r 表示插入新成员（如果原来不存在的话）；
c 表示创建新档案；
s 表示将插入的成员作为符号表保存。
编写使用静态库的代码，并链接静态库，比如 **`g++ -o main main.cpp -L. -ltest`**。

-L. 表示在当前目录搜索库文件；
-lmylib 表示链接名为 libmylib.a 的静态库。
运行可执行程序，比如 ./test。



