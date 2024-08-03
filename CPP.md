## Makefile

```makefile
#编写makefile文件
ALL:testfile   #ALL 确定最终目标文件
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

.PHONY:clean     #伪目标
clean:  
	del *.o $(TARGET) 

#其他函数
#1.过滤出不以.c结尾的字符串
object=foo.o bar.o baz.c
flitered_objects=$(filter-out %.c $object)
```

`make -f m1` 	指定文件执行make命令，可能有些makefile文件叫xxx.mk，所以需要自己指定

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







Linux系统编程和网络编程，很重要

## Linux系统编程

`more `读文件分屏显示（空格翻页）

`sudo apt-get install`：安装

`ln -s  file  file.s`：创建名字叫file.s的软连接（相当于windows下的快捷方式）文件保存的是路径，如果想要创建的软连接在任意移动，需要使用绝对路径创建。

`ln  file  file.hard`	：创建硬链接（ls -l中可以看到硬链接计数），操作系统给每个文件赋予inode，相同inode文件存在时，彼此同步，在删除时，只是改变硬链接计数，减为0时，inode结点被释放。

`stat  fileName`  :查看文件的相关属性

`chmod -u+x file.c`：a表示所有人	u表示当前用户    g表示组内其他用户	o表示不在组内的其他用户   +/-增加或删除权限	rwx权限

`chown newUser fileName` ：修改文件所属用户

`chgrp 	userGroup fileName` :修改文件所属组

`find ./balaba  -maxdepth 1 -type 'l'` :查找文件   l为软连接类型  

-type表示以文件类型查找 

-name表示以名称查找。

-maxdepth 1 表示搜索的深度，设置为1避免深入其他文件夹搜索。  

-size  +20M  -size  -50M按照文件大小范围搜索

| -xargs  与-exec作用类似，将结果集执行某一指定命令。当结果集过大时，可以进行分片映射

**注意：**xarg默认以空格分隔结果集，如果查询的文件名中有空格，就会被错误解析（文件名被分开成多个），需要加上`-print0`，将结果集用null分隔

`find  ./ -type f  -exec ls -l {} \; ` 利用`-exec或-ok`命令，将前一个语句的结果，接着执行后面的语句。`{}`是指定语法，将搜索结果放到大括号中，`;`表示语句的结束，`\`表示转义字符。

**进程管理**

`grep -r  'copy'  ./` 搜索指定目录下文件中的内容    `-r`表示递归搜索目录

`ps aux | grep  'kernel`'    `ps`搜索当前执行的进程，得到的结果集，结合 `| `管道命令在执行 `grep` 可以搜索包含指定名称的进程

`env` 查看系统环境变量

`top` 文字版任务管理器

**软件安装**

`apt-get install` 安装软件      `sudo apt-get update` 更新软件资源列表到本地，更新了才能安装软件

`sudo apt-get remove` 卸载

`sudo dpkg -i  安装包名称.deb`  通过安装包安装软件（离线下载）

`sudo aptitude show 软件名` 和apt-get作用相同，但是它可以查看软件是否安装



**压缩包管理**

`tar -zcvf 压缩包名 压缩材料`   使用gzip方式进行压缩     `tar -jcvf .. .. `   使用bzip2方式压缩   

解压缩只需要将参数中的`c`改成`x`

`zip -r 压缩包名(.zip) 压缩材料`         `unzip  压缩包名(.zip)`  可以与window操作系统下的文件互通，都有.zip



**网络管理**

`ifconfig`  查看ip



**其他命令**

`alias pg='ps aux | grep'` 给命令起别名

`umask 522` 指定用户创建的文件掩码。对于一个文件的权限有rwx rwx rwx（三个一组），对应的数字为421421421（777），设置掩码表示把对应的文件权限去掉，如522，那么就是把第一组rx，第二组w，第三组w的权限去掉，变成-w- r-x r-x。而操作系统默认刚创建的文件没有x（也就是没有执行能力）,所以最终结果为-w- r-- r--

  



**Vim**

**编辑模式**

|     按键     |                  功能                  |
| :----------: | :------------------------------------: |
|      i       |         光标位置当前处插入文字         |
|      I       |          光标所在行首插入文字          |
|   o(字母)    |       光标下一行插入文字（新行）       |
|   O(字母)    |       光标上一行插入文字（新行）       |
|      a       |          光标位置右边插入文字          |
|      A       |          光标所在行尾插入文字          |
|      s       | 删除光标后边的字符，从光标当前位置插入 |
|      S       |     删除光标所在当前行，从行首插入     |
|  :sp 文件名  |              添加横向分屏              |
| :vsp  文件名 |              添加纵向分屏              |

**移动**

| **按键** |             **功能**             |
| :------: | :------------------------------: |
| Ctrl + f |         向前滚动一个屏幕         |
| Ctrl + b |         向后滚动一个屏幕         |
|    gg    |         到文件第一行行首         |
| G(大写)  | 到文件最后一行行首，G必须为大写  |
| mG或mgg  |      到指定行，m为目标行数       |
| 0(数字)  | 光标移到到行首（第一个字符位置） |

**删除**

| **按键** |                           **功能**                           |
| :------: | :----------------------------------------------------------: |
|   [n]x   |                     删除光标后 n 个字符                      |
|   [n]X   |                     删除光标前 n 个字符                      |
|    D     |                删除光标所在开始到此行尾的字符                |
|  [n]dd   | 删除从当前行开始的 n 行（准确来讲，是剪切，剪切不粘贴即为删除） |
|    dG    |              删除光标所在开始到文件尾的所有字符              |

**复制粘贴**

| **按键** |           **功能**           |
| :------: | :--------------------------: |
|  [n]yy   |   复制从当前行开始的 n 行    |
|    p     | 把粘贴板上的内容插入到当前行 |

**查找**

| **按键** |                  **功能**                  |
| :------: | :----------------------------------------: |
| /字符串  | 从当前光标位置向下查找（n，N查找内容切换） |
| ?字符串  | 从当前光标位置向上查找（n，N查找内容切换） |

**可视模式**

| **按键**  |                           **功能**                           |
| :-------: | :----------------------------------------------------------: |
|     v     | 按字符移动，选中文本，可配合h、j、k、l选择内容，使用d删除，使用y复制 |
| Shift + v | 行选（以行为单位）选中文本，可配合h、j、k、l选择内容，使用d删除，使用y复制 |
| Ctrl + v  | 列选 选中文本，可配合h、j、k、l选择内容，使用d删除，使用y复制 |

**撤销**

| **按键** |      **功能**      |
| :------: | :----------------: |
|    u     |        撤销        |
|  ctrl-r  | 反撤销（撤销撤销） |



---



**gcc编译**

预处理：展开宏、头文件、替换条件编译、删除注释空行空白（hello.i）  `gcc -E`

编译：检查语法规范（hello.s）  `gcc -S` 

汇编：将汇编指令翻译成机器指令（hello.o）   `gcc -c`

链接：数据段合并、数据地址回填（a.out）`gcc 无参数` 

编译阶段消耗的时间和系统资源最多！

`gcc -I./include -o hello` 当头文件和源文件不在同一个目录下，需要指定头文件目录

`-g` 编译时添加调试语句。主要支持gdb调试

`-Wall` 显示警告

 `-D`  向程序中动态注册宏定义

`-l` 指定动态库名

`-L` 指定动态库路径

**制作静态库**

`ar rcs lib库名.a  add.o sub.o div.o`  规定静态库名称以`lib`开头`.a`结尾

`gcc test.c lib库名.a -o a.out` 将静态库编译到可执行文件中

**制作动态库**

`gcc  -c test.c -o test.o -fPIC`  将.c生成 .o（生成与位置无关的代码）

`gcc -shared  -o lib库名.so  add.o sub.o`  制作动态库

`gcc test.c -o a.out -lmymath -L ./lib`编译可执行程序时，指定所使用的动态库。`-l指定库名` `-L`指定库路径

`ldd a.out`  该命令可以查看可执行文件动态加载了哪些动态库

==注意：会出错！！==

**原因：**这两个没有关系。链接器：工作于链接阶段，工作时需要`-l 和-L `。动态链接器：工作于程序运行阶段，工作时需要提供动态库所在目录位置。一般去固定地方寻找动态库，所以我们需要将自己的动态库放到对应位置。命令：`export  LD_LIBRARY_PATH=动态库路径`（修改是临时的！每次重启终端就无效了）。**永久修改方式：**需要写入终端配置文件，家目录下的隐藏文件`.bashrc` 写入`export LD LIBRARY_PATH=动态库路径` 保存后`source .bashrc`让配置生效。





**gdb调试工具**

`gcc -g` 使用该参数编译可执行文件，得到调试表

`list ` 简写为`l` 列出源码，根据源码指定行号设置断点

`break/b 20`  在20行设置断点    

`run/r` 运行程序

`next/n` 下一条指令，会越过函数

`step/s` 下一条指令，进入函数

`print/p 变量名` 查看变量的值

`continue` 继续执行断点后续指令

`quit` 退出当前调试



`start` 不加断点，一步一步执行看看

`finish` 结束当前函数

`set args` 设置main函数命令行参数

`run 字串1 字串2...` 设置main函数命令行参数

`info b` 查看断点信息

`b 20 if i=6` 设置条件断点

`ptype i` 查看变量类型，一般查看自定义类型。

`backtrace/bt` 查看栈帧调用（函数调用）的层级关系

`frame 1` 根据栈帧编号，切换栈帧。一般和bt结合使用

 `display` 设置跟踪变量，把每次都把变量值打印出来。  

`undisplay  1` 根据变量编号删除跟踪变量。

gdb进去之后直接run，可以直接找到段错误的位置，重要有点用了！！

  





