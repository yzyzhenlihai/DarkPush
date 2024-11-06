#语法

## 类型转换关键字

1. `static_cast`  用于基本数据类型和类层次中的**安全转换**，通常用于类型之间的已知转换，不涉及运行时检查。

2. `dynamic_cast` 主要用于类层次的**下行转换**（父类转换为子类，可能会导致类型未定义），提供运行时类型检查，确保转换安全性，常用于多态环境。出错会返回空指针`nullptr`

3. `const_cast` 用于移除或添加 `const`/`volatile` 修饰符，必须谨慎使用，不能修改真正的常量。

4. `reinterpret_cast` 用于底层的、低级别的强制转换，不进行类型检查，应避免不必要的使用，因其可能导致未定义行为。比如定义回调函数，需要传入`void*` 类型的参数。主要用于将**一个指针类型转换为另一种指针类型**或将**整数转换为指针**。它并不适合用于基本数据类型的转换。

   ```c++
   //uintptr_t是一个足够大的无符号整数类型，足以存放指针值，避免转换时造成数据丢失
   uintptr_t t=static_cast<uintptr_t>(i);
   task.arg=reinterpret_cast<void*>(t);
   //将void* 还原为int 时，也要先转换为uintptr
   ```



## 智能指针

1. `std::shared_ptr`  shared_ptr顾名思义是多个指针指向一块内存,被管理对象有一个**引用计数**，这个计数记录在每个指针上，几个shared_ptr指向它，这个数字就是几，当没有任何shared_ptr指向它时，引用计数为0，这时，自动释放对象。但是这里会出现一个问题，如下面的代码所示:

   ```c++
   int main(int argc, const char * argv[]) {
       auto p1 = new Test; // 划分堆空间
       std::shared_ptr<Test> sp(p1); // 创建智能指针
       std::shared_ptr<Test> sp2(p1); // 创建另一个智能指针
       return 0;
   }
   ```

   这段程序会抛出异常 **double free detected**，此处用了两个智能指针管理同一块内存，因为sp 和sp2不知道彼此的存在，所以也会重复释放，正确的做法是用已经创建的智能指针sp再去初始化sp2，让它们处于同一套内存管理体系。

   所以这里就引入了`std::make_shared<T>(args)` 去创建对象，而不是用`new`，这样就可以防止我们去使用原始指针创建多个引用计数体系。

2. 





## 容器

### vector

`resize(size,value)`  使用resize进行扩容时，元素之间的相对位置并不会发生变化。但是可能为由于**内部内存管理**，会将数据重新复制到一片新开辟的地址空间。



##并行

`std::atomic`  对变量提供原子操作，避免使用`std::mutex`来造成性能开销和复杂性

1. **`etch_add`：**

   - 原子地增加一个值，并返回增加前的值。适合在需要获取旧值的场景。

   ```
   userCount.fetch_add(1);
   ```

2. **`fetch_sub`：**

   - 原子地减少一个值，并返回减少前的值。

   ```
   userCount.fetch_sub(1);
   ```

3. **`store`：**

   - 原子地设置一个新值。这对于直接更新变量是非常有效的。

   ```
   userCount.store(newValue);
   ```

4. **`load`：**

   - 原子地读取当前值。这在需要安全地读取变量时使用。

   ```
   int currentCount = userCount.load();
   ```

5. **`exchange`：**

   - 原子地设置一个新值，并返回旧值。

   ```
   int oldValue = userCount.exchange(newValue);
   ```



`std::unique_lock<std::mutex> locker(pool->mtx);`   

RAII语法的模板类，保证了所有栈对象在声明周期结束时回被销毁，会自动调用`unlock()` 。并且可以显式的调用lock和unlock。如果用到了`std::condition_variable::wait`，则必须用`std::unique_lock`

##函数对象包装

\#include\<functional\>

`std::bind() `  当你调用类的成员函数时，**隐式地**会将对象指针（即 `this` 指针）传递给该函数。这使得**成员函数能够访问该对象的成员变量和其他成员函数**。在使用 `std::bind` 或其他类似的方式时，你需要显式提供对象指针，以确保函数能够正确绑定到特定对象上。



`explicit` 关键字在 C++ 中用于防止编译器进行隐式类型转换。它主要应用于构造函数和转换运算符。

```c++
class MyClass {
public:
    explicit MyClass(int value) {
        // 构造函数实现
    }
};
// 没有使用 explicit 时，以下代码会自动转换
MyClass obj = 42; // 隐式转换，会调用 MyClass(int)
// 使用 explicit 时，以下代码会报错
MyClass obj2 = MyClass(42); // 必须显式调用构造函数
```



##右值引用

`std::move` 将左值转换为右值引用。可以避免不必要的复制操作，特别是当任务对象是较大的或复杂的类型（比如使用了大量资源的对象）时。通过移动，资源的所有权从源对象转移到目标对象，而不是创建一个新副本。

`std::forward<type> task ` 将task进行完美转发，保持其原始的值类别（左值或右值）。这样可以避免不必要的拷贝，提高性能。



## 类

###类的定义和实现

如果你将类的定义和实现都放在头文件中，那么每次在源文件中包含该头文件时，编译器都会看到类的完整定义和成员函数的实现。这会导致一些特定的问题和影响：

**编译时间变长**：每次包含头文件时，编译器都会重新编译类的实现。如果多个源文件都包含了相同的头文件并实例化了类，类的成员函数会在每个源文件中都被编译，这会导致重复编译，增加编译时间。

**代码重复**：每个包含头文件的源文件都会生成该类的成员函数的实现代码，这可能导致多个目标文件中包含相同的代码，最终在链接阶段可能出现重复符号。

**模板类和内联函数**：有时候类的成员函数是模板函数或者内联函数，它们必须在头文件中定义和实现。模板类成员函数在编译时需要实例化，因此必须在头文件中提供实现。这是一个特定的例外，不一定适用于所有类。



## 函数

`std::search`  下面是原型

template< class ForwardIt, class T >

 ForwardIt search( ForwardIt first, ForwardIt last, const T& value );

是一个非常强大的函数，可以帮助你在一个序列中查找一个**单一的元素**或者一个**子序列**。在\<algorithmn\>中



## 正则表达式

\#include\<regex\>

`std::regex_match(text,pattern)`

 是C++标准库中用于执行完全匹配的正则表达式函数。它用于判断一个字符串是否**完全与给定的正则表达式模式匹配。**

`std::regex_search(text,pattern)` 

是C++标准库中用于在字符串中搜索匹配的正则表达式模式的函数。与 `std::regex_match` 不同，`std::regex_search` 用于查找字符串中的**部分匹配**，而不是要求整个字符串完全匹配。

`std::regex_replace(text,pattern,replacement)` 

是C++标准库中的函数，用于替换字符串中与正则表达式模式匹配的部分。允许将匹配的文本替换为指定的内容。

`std::smatch` 

是C++标准库中的一个类，用于**存储正则表达式的匹配结果**。它通常用于捕获、存储和访问正则表达式中的匹配子串。

当匹配成功时，`matches[0]` 包含整个匹配的字符串，而 `matches[1]` 包含**捕获组** 1 中的内容。正则表达式一个括号是一个捕获组

```c++
#include <iostream>
#include <regex>
#include <string>
int main() {
    std::string text = "My phone number is (123) 456-7890.";
    std::regex pattern(R"(\(\d{3}\) \d{3}-\d{4})");
    std::smatch matches;
    if (std::regex_search(text, matches, pattern)) {
        std::cout << "Match found: " << matches[0] << std::endl;
        std::cout << "Capture group 1: " << matches[1] << std::endl;
    } else {
        std::cout << "No match found." << std::endl;
    }
    return 0;
}
```



# 设计思想





