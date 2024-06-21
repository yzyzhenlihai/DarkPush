#c++做题

1. <iomanip>使用setprecision(n)可控制输出流显示**浮点数的数字**个数,setprecision(n)与setiosflags(ios::fixed)，setprecision(n)与setiosflags(ios::fixed)**合用**，可以控制小数点右边的数字个数，可以控制小数点右边的数字个数

2. setw(2) 控制输出整型的长度，setfill('0')，左边补零，两者共同使用控制输出。

3. reverse(a,a+10)函数，反转数组。   <algorithm>

4. sort(a,a+10)  排序函数    <algorithm>

5. 万能头文件：<bits/stdc++.h>

6. memset(a,0,sizeof(a))，将数组a初始化为0的函数     头文件<cstring>

7. 判断质数的方式：

   1. 偶数位数回文数（除了11）必定不是质数
   2. 偶数肯定不是质数

8. int的范围   -32768~32767

   long long 的范围   -2147483648~2147483647

9. map函数：dist.count(x) == 1;*//x元素存在*   dist.count(x) == 0;*//x元素不存在*

10. vector函数： 

    a.empty(); //判断a是否为空，空则返回ture,不空则返回false

    a.back(); //返回a的最后一个元素

    a.front(); //返回a的第一个元素

    a.pop_back(); //删除a向量的最后一个元素

    a.push_back(x) //在向量最后位置插入



10. queue函数

    push() 在队尾插入一个元素

    pop() 删除队列第一个元素

    size() 返回队列中元素个数

    empty() 如果队列空则返回true

    front() 返回队列中的第一个元素

    back() 返回队列中最后一个元素

11. <algorithm>头文件 :  max()、min()、abs()、

12. typedef pair<int,int> PII   可以用来存储像坐标一样的数据类型

13. memset(dis,-1,sizeof(dis))  用来初始化数组     <cstring>头文件中

    memset是按字节进行赋值的，所以memset(f，0x3f，sizeof f)，就相当于给4字节的int赋值为0x3f3f3f3f。

14. 在设置最大值时，可以使用    INF=0x3f3f3f3f  是一个很好的选择

    memset(a,INF,sizeof(a))。//通过这种方式，将数组初始化为无穷大

15. set函数：

    insert()   //插入元素

    erase()   //删除元素

    count()  // 判断元素是否存在

    size()    //获取元素个数

    clear()   //清空集合

    

# java



super关键字必须是子类构造方法中的第一条语句

子类不继承父类的构造方法！！！

接口不允许继承抽象类，但允许一个接口继承多个接口

JFrame 缺省布局使用的是边界布局管理器(BorderLayout).

JPanel 缺省布局使用的是流式布局管理器(FlowLayout).

JFrame是JWindow类的直接子类

##javaAPI

###Object类

每个类都直接或间接继承Object类，所有类型的基类

```java
String toString();//返回对象的字符串表示形式
int hashCode();//返回对象的散列码值
boolean equals();//判断两个对象是否相等
```



###String类

创建对象后是个常量，内容和长度不可改变，如果对一个字符串进行修改需要创建一个新的字符串

```java
char charAt(int index);//返回位置上的字符
boolean equals(String string);//字符串比较
String toUpperCase();//将字符串大写
static String valueOf(int i);//将整型转换为字符串
char[] toCharArray(String str);//将字符串转化为字符数组
byte[] getBytes();//将字符串转化为字节数组
```

###StringBuffer类

是一个字符串缓冲区，内容和长度可以改变，增添和删除不会产生新对象

**线程安全**的，不能被同步访问

```java
StringBuffer append(char c);//添加,也可以添加字符串
StringBuffer insert(int index,String str);//插入

StringBuffer deleteCharAt(int index);//删除
StringBuffer delete(int start,int end);//删除指定范围的字符串

void setCharAt(int index,char ch);//修改指定位置的字符序列
StringBuffer replace(int start,int end,String s);//替换指定范围的字符串

String toString();//返回StringBuffer缓冲区中的字符串
StringBuffer reverse();//将字符串反转
```

###StringBuilder类  

**非线程安全**的，可以被同步访问。操作与StringBuffer类似，但**效率最高**



###System类

属性和方法都是静态的，所以可以直接用类名调用

```java
static void currentTimeMills();//返回以毫秒为单位的当前时间
//将源数组某个数开始复制到目标数组的某个位置，长度为length2
static void arraycopy(Object src,int srcPos,Object dest,int destPos,int length);
```

###Thread类

```java
static void sleep(long millis);//使程序休眠millis毫秒，静态方法
```

### Random类

import java.util.Random

```java
Random Random();//构造方法，不传入种子，默认以系统时间作为种子
Random Random(long seed);//具有相同的种子，会产生相同的序列

double nextDouble();   
float nextFloat();
int nextInt();
int nextInt(int n);//产生[0,n)的随机数
```





##集合

import  **java.util ** 包中

###Collection集合     单列集合

 **Iterator接口**

```java
//迭代器的索引位于第一个元素之前，不指向任何元素
Iterator it=list.iterator();//获得对象的迭代器
void remove();//利用迭代器的方法删除集合中的某个数据
void hasNext();//判断集合中是否存在下一个元素
void Next();//将迭代器的索引向后移一位，并获取迭代器指代的元素 ，
			//迭代器获得的元素都是Object类型，如果想获得特定类型，需要进行强制转换

//迭代删除元素
正确：it.remove()//利用删除元素的迭代器来删除元素,remove为迭代器对象的方法
错误：list.remove(obj)
```

**List接口**

```java
//ArrayList集合

```



**set接口**

```java
/*add()添加元素时，为了确保不重复的元素 
1.先获取当前存入对象的hashCode()的散列值 
2.通过散列值计算出一个存储位置 
3.判断是否有对象存在，如果存在就舍弃，如果不存在就存入集合*/
HashSet集合  //内部元素无序
TreeSet集合  //可以对集合中的元素进行排序
//自然排序
自定义类需要继承Comparable接口，并重写接口中的 compareTo()方法
public int compareTo(Student o){
    return 0;//集合中只有一个元素
    return 1;//按照怎么存怎么取
    return -1;//按照存入元素的到序进行存储
}
//另一中排序方式--比较器排序
实现 Comparator()接口中的compare()方法和equals()方法
```

###Map集合 双列集合

```java
//HashMap
Set keySet()    //获得映射中包含键的Set视图
Collection<V> values()  //返回映射中包含的值的Collection视图 Collection是所有单列集合的父接口
Set<Map.Entry<K,V>>entrySet();//获得映射中键值对的集合
boolean containsKey(Object key) //判断是否包含某个键
boolean containsValue(Object value) //判断是否包含某个值
```

```java
//遍历Map

//第一种方式   先获取key的集合，通过遍历key集合获得value
		Set keySet=map.keySet();
		Iterator it=keySet.iterator();
		while(it.hasNext()) {
			Object key=it.next();
			Object value=map.get(key);
			System.out.println("key:"+key+"    value:"+value);
		}
//第二种方式
		Set entrySet=map.entrySet();
		Iterator it=entrySet.iterator();
		while(it.hasNext()) {
            //每个Map.Entry对象代表Map中的一个键值对(Entry是Map内部接口)
			Map.Entry entry=(Map.Entry)(it.next()); 
			Object key=entry.getKey();
			Object value=entry.getValue();
			System.out.println("key:"+key+"     value:"+value);
		}
	}
```

**Properties集合**

```java
//Properties集合可以用来存取应用的配置项
//主要用来存储字符串类型的键和值

		Properties p=new Properties();  
		//存字符串
		p.setProperty("1", "jack"); 
		p.setProperty("2", "lucy");
		p.setProperty("3", "mike");
		//获取Enumeration对象所有键的枚举
		Enumeration names=p.propertyNames();
		//循环遍历所有键
		//hasMoreElements()：判断Enumeration对象中是否还有数据
		while(names.hasMoreElements()) {
            //nextElement()：获取Enumeration对象中的下一个数据
			String key=(String) names.nextElement();
            //获得对应键的值
			String value=p.getProperty(key);
			System.out.println("key:"+key+"     value:"+value);
		}
```



**foreach循环**

不能修改集合中的元素

```java
for(Object obj : List);//Object 是集合中的元素类型
for(String str : strs);//访问字符串集合中的元素
```



**instanceof运算符**

用于强制转换前判断对象的真实类型，以免转换异常



## 文件

**import  java.io.* ;**

###File类

```java
File file=new File("src/yzy.txt");//相对路径
		try {
			if(file.exists()) {
				file.delete();
			}else {、
                //创建新文件
				System.out.println(file.createNewFile());
			}
		}catch(IOException e) {
			e.printStackTrace();
		}
```

###字节流   

顶级父类是**InputStream类**和**OutputStream类**

```java
int read();//读取一个字节
int read(byte[] b);//读取若干个字节，返回字节数
void close();//关闭输入流并释放与该流关联的所有系统资源
```

```java
//读取文件
		FileInputStream input=null;
		try {
            //在项目根目录下的文件
			input=new FileInputStream("yzy.txt");
			while(true) {
				//将一字节转换为int类型
				int b=input.read();//一次读取一个字节
				if(b==-1) {
					break;
				}else {
					System.out.print((char)b);
					
				}
			}
            
		}catch(IOException e) {
			e.printStackTrace();
		}finally {
			//关闭字节输入流
			if(input!=null) {
				input.close();
			}
		}
```



```java
void write(int b);//向输出流写入一个字节
void write(byte[] b);//把字节数组中的字节全部写入输出流
```

```java
//写文件
		FileOutputStream output=null;
		try {
			output=new FileOutputStream("yzy.txt",true);//true表示在文件中追加内容
			String str="牛牛牛";
			byte[] b=str.getBytes();//将字符串转化为字节数组
			for(int i=0;i<b.length;i++) {
				output.write(b[i]);
			}
		}catch(IOException e) {
			e.printStackTrace();
		}finally {
			if(output!=null) {
				output.close();
			}
		}
```

###字符流

```java
		//利用包装流提高读写数据的效率
		BufferedReader br=null;
		BufferedWriter bw=null;
		try {
			FileReader reader=new FileReader("yzy.txt");
			//创建一个BufferedReader缓冲对象,相当于赋予字符流对象缓冲区
			br=new BufferedReader(reader);
			//在文件后面添加内容
			FileWriter writer=new FileWriter("new.txt",true);
			//创建一个BufferedWriter缓冲对象
			bw=new BufferedWriter(writer);
			String str;
			//每次读取一行文本，判断是否到末尾
            //readLine() 用来一次读取一行文本
			while((str=br.readLine())!=null) {
				bw.write(str);
				bw.write("\n");
			}
		}catch(IOException e) {
			e.printStackTrace();
		}finally {
			br.close();
			bw.close();
		}
```





##JDBC程序（连接数据库）



```java
package cn.itcast.java_exam;
import java.sql.*;
import java.lang.Class;
public class java_question2 {

	
	public static void main(String[] args)throws SQLException {
		Connection conn=null;
		Statement stmt=null;
		ResultSet result=null;
		try {
			//注册数据库的驱动
			Class.forName("com.mysql.cj.jdbc.Driver");
			//建立数据库连接
			String url="jdbc:mysql://localhost:3306/studentbase";
			String user="root";
			String password="yu121314";
			conn=DriverManager.getConnection(url,user,password);
			//获得statement对象
			stmt=conn.createStatement();
			//操作结果集
			String sql="select* from course";
			result=stmt.executeQuery(sql);
            
           	//遍历结果集（一个关系表）
			while(result.next()) {
				int cno=result.getInt("cno");
				String cname=result.getString("cname");
				int cpno=result.getInt("cpno");
				int ccredit=result.getInt("ccredit");
				System.out.println("cno:"+cno+" cname:"+cname+" cpno:"+cpno+" ccredit:"+ccredit);
			}
		}catch(ClassNotFoundException e) {
			e.printStackTrace();
		}finally {
            //释放ResultSet结果集对象
			if(result!=null) {
				try {
					result.close();
				}catch(SQLException e) {
					e.printStackTrace();
				}
				result=null;
			}
            //释放Statemen对象
			if(stmt!=null) {
				try {
					stmt.close();
				}catch(SQLException e) {
					e.printStackTrace();
				}
				stmt=null;
			
			}
            //释放Connection对象
			if(conn!=null) {
				try {
					conn.close();
				}catch(SQLException e) {
					e.printStackTrace();
				}
				conn=null;
			}
		
		}
	
	}
}

```



##GUI(用户图形界面)

# javaweb

正常的cookie只能在一个应用中共享，即一个cookie只能由创建它的应用获得。

cookie.setPath(“/”)

**获得请求消息行的方法**

request.getContextPath();获得URL中属于web应用程序的路径，例如  /chapter05 （chapter05为web应用的根目录）

request.getRequestURI()//获得主机号和端口之后，参数部分之前的部分

 request.getServletPath()//获取Servlet的名称或Servlet所映射的路径

**获得请求参数**

**getAttribute(String name)**取得 name 属 性 的 值 时 ，它 会 回 传 一 个java.lang.Object，因此，我们还必须根据name属性值的类型做**转换类型(Casting)**的工作。

request.getAttribute("")；//在获得属性值之后需要进行类型的强制转换，否则编译会不通过



**获取网页连接信息**

request.getRemoteAddr()//获得服务器IP地址



**<jsp: include>和include指令的区别**：< jsp:include>标签是动态引入， < jsp:include>标签涉及到的2个JSP页面会被翻译成2个servlet，这2个servlet的内容在执行时进行合并。而include指令是静态引入，涉及到的2个JSP页面会被翻译成一个servlet，其内容是在源文件级别进行合并。

# Python

str.isdigit()  //判断字符串是不是由数字组成

print(" ",%( ) )   //格式化输出

print(" ",end=' ')    //end=' '表示以' '结尾    ,不加end的话默认是换行

str()   //强转为字符串

find(子串，start，end)  //查找字符串中是否包含某个数，返回值为在字符串中的位置，没有返回-1

count(子串，start,end)  //查找子串出现的次数

rstrip()  //删除字符串右边的空白

split()   // 以空格分隔字符串

center(width,"填充字符")    

str.replace(old,new)    //将字符串中的字符或字符串替换成新的字符串 

str.capitalize()             //将字符串开头大写，其余部分变为小写

str.format(a,b)     //字符串格式化    将变量按顺序填入到字符串中的{}中

in 是成员操作符，可以对序列（字符串、元组、列表、集合）做判断成员是否在其中

isdisjoint()  函数用于判断两个集合是否有相同的元素

空集合只能通过set()创建

单引号里不能再有单引号

os模块 chdir 方法来改变默认目录   getcmd获得当前目录

匿名函数使用 lambda关键字定义     //在定义是需要定义一个变量调用该表达式  

```python
result=lambda r,p:r*r*p
print(result(2,3))
```

查看关键字列表方法   keyword.kwlist

\__mro__ 可以查看方法搜索顺序

Python不允许Tab键和空格混合使用

//整除           /除  结果是浮点型

定义函数时指定了默认值的参数（关键字参数）必须在**没有默认值**的参数之后

p=list.index(value)    获得某个元素的下标

**zip()**  函数用于将**可迭代**的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些**元组组成的列表**。

from collections import Counter     **Counter**用于统计可迭代对象中元素出现的次数，返回一个字典



## Python方法

1. **类实例方法**

   通过对象调用，传参的第一个参数默认为self

2. **类方法**

   通过类调用，传参的第一个参数默认为cls，用@classmethod修饰

   ```python
   @classmethod
   def info(cls):
   	print("正在调用类方法")
   ```

   

3. **类静态方法**

   既可以使用类调用，也可以使用实例调用。无法调用任何类属性和类方法

   没有像self、cls这样特殊的参数，需要用＠staticmethod修饰



##文件读取

with open("文件名")  as  文件对象名 ：

​	contents=file.**read()**   //read() 方法读取文件的全部内容，并在文件末尾是返回一个空字符串，显示出来是个空行,

​		                                  //支持读取**指定字节**的内容

readlines()   //读取文件每一行并存在一个列表中

readline()     // 之有在一组打开与关闭操作之间，readline()才会逐行读取。若重新打开文件，readline()将会从第一行开始读取。

##文件写入

在 open中提供另一个实参

读取模式 ‘r’     写入模式 ‘w’     附加模式  ‘a’    读写模式  ‘r+’       

write()  //该方法用于写入文件时使用

python只能将**字符串**写入文本文件，存取**数值数据**是需要str()转换为字符串格式



##异常

**BaseException** —— 所有异常的基类

**Exception** —— 常规错误的基类

当用户定义的条件不满足时，它会触发**AssertionError**异常

**try-except-else 代码块**

try中书写可能引发异常的语句，except中书写产生异常后需要进行的处理，else中书写语句正常执行的代码

except没有指明异常，就不会捕获任何异常

在except中添加**pass**语句，表示发生异常时**静默**，但程序还是照常执行



**raise语句**：主动抛出异常，在try中被捕获由except处理

当在没有引发过异常的程序使用**无参**的 raise 语句时，它默认引发的是 RuntimeError 异常

或者是引发上一个异常

```python
try:
    a = input("输入一个数：")
    #判断用户输入的是否为数字
    if(not a.isdigit()):
        raise ValueError("a 必须是数字")
except ValueError as e:
    print(e)
```

**自定义异常类**

```python
#Exception是基类
class MyException(Exception):
    def __init__(self,string):
        super().__init__(self)   	#调用基类的构造函数
        self.string=string       	
    def __str__(self):              
        return self.string
#抛出异常   
raise MyException("输入内容必须为整数")
print(e)
```



**捕获异常**

1. 使用 sys 模块中的 exc_info 方法；
2. 使用 traceback 模块中的相关函数。

```python
#exc_info() 方法会将当前的异常信息以元组的形式返回，该元组中包含 3 个元素，分别为 type、value 和 traceback
print(sys.exc_info())

 # 捕捉异常，并将异常传播信息输出控制台
 traceback.print_exc()
 # 捕捉异常，并将异常传播信息输出指定文件中
 traceback.print_exc(file=open('log.txt', 'a'))
```





##CSV库



##json模块

**import json**

json是一种轻量级数据交换格式

json.dump("数据"，"文件对象")    //用来将python转化为json类型 并写入文件

json.load("文件对象")                      //加载json中的信息到内存中



##pyinstaller库

**import pyinstaller**

 **pyinstaller -F 测试.py**    cmd中打包py文件，生成可执行文件

-f 指定生成单独的exe文件(放在dis文件夹中)

-w 表示程序后台运行，没有窗口 （仅对 Windows 有效）

-c 表示提供程序视窗    （仅对 Windows 有效）

## python画图

1）figure相当于大画布，整个作图的大环境
2）axes相当于是小画布，一个figure上能有很多的axes（通过subplot 来设置他们的排列顺序）
3）经典问题：plt.plot和ax.plot的区别
相比于前者，后者相当于是线生产出figure 再生产出axes画了一个图。plt.plot()通过plt.gca()获得当前的Axes，再调用ax.plot()方法实现绘图。然这句话读起来确实有一些拗口，就这么说吧，真的要标准的画图，用ax.plot肯定是要标准一些的。 我称之其为《最标准的过程》

```python
#二维图

x=np.arange(1,10,1)
y= x**2

Ix=[random.randrange(0,100) for i in range(100)]
Iy=[random.randrange(0,100) for i in range(100)]
#：plt.subplots（）返回 类似的 [ [figure], [axes1, axes2…] ]类似的东西
fig,axes=plt.subplots(2,3,figsize = (15, 10))  #这个参数默认的情况下是1

#通过索引访问不同的子图
axes[0][0].plot(x,y,label='抛物线')
axes[0][0].set_title('抛物线') #设置子图标题
axes[0][0].set_xlabel('x') #设置横坐标名称
axes[0][0].set_ylabel('y') #设置纵坐标名称
#设置图例
axes[0][0].legend(loc='best'，bbox_tox_anchor=(num1,num2))	#num1控制左右移动，num2控制上下移动  
#设置坐标轴范围
axs[0][0].set_xlim(min,max) , axs[0][0].set_ylim(min,max)
#画散点图
axes[1][1].scatter(Ix,Iy)
#设置主标题 fontsize设置字体大小
plt.suptitle('标题名',fontsize=15)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.2)  # 调整子图间距
plt.show()

```

```python
#画三维图
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']  #用来显示正确的中文图例
plt.rcParams['axes.unicode_minus']=False    # 默认是使用Unicode负号，设置正常显示字符，如正常显示负号
fig=plt.figure()
ax=plt.axes(projection='3d')
#限定x,y轴的范围
xx=np.arange(-3,3,0.01)
yy=np.arange(-3,3,0.01)
#生成网格点的坐标矩阵
X,Y=np.meshgrid(xx,yy)
Z=(X+Y)**2+np.sin(Y)

plt.xlabel('x轴')
plt.ylabel('y轴')

ax.plot_surface(X,Y,Z)
plt.show()

```

```python

'''
绘制饼图
[explode]：默认值为None的可选参数。若非None，则是和x相同长度的数组，用来指定每部分的离心偏移量。	就是突出显示
[labels]：列表，指定每个饼块的名称，默认值None，为可选参数。
[colors]：特定字符或数组，指定饼图的颜色，默认值None，为可选参数。

[autopct]：特定字符，指定饼图中数据标签的显示方式，默认值None，为可选参数。	设置数据小数点位数
[pctdistance]：浮点数，指定显示比例距离圆心的距离。默认值0.6，为可选参数。
[labeldistance]：浮点数，指定每个扇形对应标签与圆心的距离，默认值1.1，为可选参数。
[startangle]：浮点数，指定从x轴逆时针旋转饼图的开始角度，默认值None(0度)，为可选参数 
[radius]：浮点数，指定饼图的半径，默认值1，为可选参数。 
[textprops]：字典，设置文本对象的字典参数，默认值None，为可选参数。 
**kwargs：不定长关键字参数，用字典形式设置条形图的其它参数。

'''
plt.pie(x, [explode], [labels], [colors], [autopct], [pctdistance],  [labeldistance], [startangle], [radius], [textprops], **kwargs)

plt.pie(x=[smoke,no_smoke],labels=['吸烟','不吸烟'],autopct='%.2f%%',colors=["#d5695d", "#5d8ca8"])

#保存图片
 plt.savefig('相对路径或绝对路径',dpi=300)  #dpi为像素点

'''
绘制箱型图
x：指定要绘制箱线图的数据，可以是一组数据也可以是多组数据；
notch：是否以凹口的形式展现箱线图，默认非凹口；
sym：指定异常点的形状，默认为蓝色的+号显示；
vert：是否需要将箱线图垂直摆放，默认垂直摆放；
whis：指定上下须与上下四分位的距离，默认为1.5倍的四分位差；
positions：指定箱线图的位置，默认为range(1, N+1)，N为箱线图的数量；
widths：指定箱线图的宽度，默认为0.5；
patch_artist：是否填充箱体的颜色，默认为False；
meanline：是否用线的形式表示均值，默认用点来表示；
showmeans：是否显示均值，默认不显示；
showcaps：是否显示箱线图顶端和末端的两条线，默认显示；
showbox：是否显示箱线图的箱体，默认显示；
showfliers：是否显示异常值，默认显示；
boxprops：设置箱体的属性，如边框色，填充色等；
labels：为箱线图添加标签，类似于图例的作用；
filerprops：设置异常值的属性，如异常点的形状、大小、填充色等；
medianprops：设置中位数的属性，如线的类型、粗细等；
meanprops：设置均值的属性，如点的大小、颜色等；
capprops：设置箱线图顶端和末端线条的属性，如颜色、粗细等；
whiskerprops：设置须的属性，如颜色、粗细、线的类型等；
manage_ticks：是否自适应标签位置，默认为True；
autorange：是否自动调整范围，默认为False；
'''
axes[0].boxplot(df['每月酱类食用量（g）'])

'''
聚类谱系图绘制
from scipy.cluster.hierarchy import linkage, dendrogram
Z：层次聚类的结果，即通过scipy.cluster.hierarchy.linkage()函数计算得到的链接矩阵。

p：要显示的截取高度（y轴的阈值），可以用于确定划分群集的横线位置。

truncate_mode：指定截取模式。默认为None，表示不截取，可以选择 'lastp' 或 'mlab' 来截取显示。

labels：数据点的标签，以列表形式提供。

leaf_font_size：叶节点的字体大小。

leaf_rotation：叶节点的旋转角度。

show_leaf_counts：是否显示叶节点的数量。

show_contracted：是否显示合并的群集。

color_threshold：显示不同颜色的阈值，用于将不同群集算法聚类为不同颜色。

above_threshold_color：超过阈值的线段颜色。

orientation：图形的方向，可以选择 'top'、'bottom'、'left' 或 'right'。

'''
```



## 数据分析

```python
#修改指定列名,利用inplace=True可直接在原数据框内改动
df.rename(columns={'购药时间':'销售时间'},inplace=True)

#判断记录是否有重复值
#subset:以哪几列作为基准列，判断是否重复，如果不写则默认所有列都要重复才算
#keep: 保留哪一个，fist-保留首次出现的，last-保留最后出现的，False-重复的一个都不保留，默认为first
#inplace: 是否进行替换，最好选择False，保留原始数据，默认也是False

df.duplicated(subset=['列名'],keep='first',inplace=True)   #返回布尔型数据，告诉重复值的位置
df[df.duplicated()==True]#打印重复值
df.drop_duplicates(inplace=True)#删除重复行
df.reset_index()			#删除完行后需要重置索引

#缺失值处理
#分析表格数据，可以获得列的空值情况
df.info()
print(df.isnull().any())  #可以获得对应列是否有空值，df.isnull()返回所有单元格内是否为空的情况
#1.删除缺失值
#dropna(axis,subset,how,thresh,inplace)
#axis: 删除行还是列，行是0或index,列是1或column，默认是行
#subst: 删除某几列的缺失值，可选，默认为所有列
#how: any or all，any表明只要出现1个就删除，all表示所有列均为na才删
#thresh: 缺失值的数量标准，达到这个阈值才会删除
#inplace: 是否替换
df.dropna(how='any', axis=0,inplace=True)#表示只要有一列有空就删除

#2.填充缺失值
#fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
#value：固定值，可以用固定数字、均值、中位数、众数等，此外还可以用字典，series等形式数据；
#method:填充方法，有ffill（用前一个填充）和bfill（用后一个填充）两
#axis: 填充方向，默认0和index，还可以填1和columns
#inplace:在原有数据上直接修改
#limit:填充个数，如1，每列只填充1个缺失值
data1.fillna({"订单号": "下次再补", "数量": "估计"})#通过传入字典的方式对不同列填充不同的值



#异常值处理
df.describe()   #描述一列的统计信息，比如均值，方差，四分位点
sales=sales[(sales['销售数量']>0)&(sales['应收金额']>0)&(sales['实收金额']>0)]#筛选对应行
```



## pandas库

```python
#Series 类似表格中的一个列（column），类似于一维数组，可以保存任何数据类型
pandas.Series( data, index, dtype, name, copy) #dtype指定数据类型,默认会自己判断

#利用字典创建Series
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
myvar = pd.Series(sites)

#DataFrame 是一个表格型的数据结构，它可以被看做由 Series 组成的字典（共同用一个索引）
pandas.DataFrame(data, index, columns, dtype, copy)

#返回指定行表格数据
df.iloc[0]   #返回第1行数据

#显示表格的前几行或后几行
df.head(10)
df.tail(2)

#获得DataFrame数据类型的行数和列数
df.shape[0]  #获得行数
df.shape[1]  #获得列数

#统计列表中每个元素的个数
result=pd.value_counts(num)

#获得narrary类型的数据
a=df.values #获得矩阵类型

#将numpy数组转化为DataFrame类型
df = pd.DataFrame(arr)

#提取其中部分列并形成新的dataframe对象
df=df[['城市序号','年商品需求量']]
df=df.iloc[0,0:4]

#修改列名
df.columns=['城市序号','年商品需求量','X坐标','Y坐标']
temp.name='判断'  #修改series类型的列名
df.rename(column={'a':'A','b':'B'}) #rename中column传入的是一个map类型

#删除某列或某行 
df.drop([行索引或行名],axis=0,inplace=False)  #axis=0删除行，axis=1删除列.默认inplace=False，原来的数据保持不变
del df['列名']

#在删除完某行后，重置索引顺序
df.reset_index(drop=True)		#drop=True表示重置索引，不把索引添加为新的列

#两个表格连接操作
df3 = pd.merge(df1, df2, on='key')#类似与关系数据库sql的连接操作
df3 = pd.concat([df1,df2],axis=1) #axis=1表示水平方向堆叠

#对表格的数据进行分段
'对年龄进行分段后构建哑变量'
bins=[0,14,35,65,200]#年龄范围
label4=['儿童','青年人','中年人','老年人']
ages=pd.cut(df['年龄'],bins=bins,labels=label4,right=True)#表示分组右边闭合，right=False表示分组左边闭合

#添加新列
df['患高血压情况']=high_blood_judge

#按某列条件筛选出对应行
#可以定义匿名函数，也可以自定义函数
df_young = df[df['年龄'].map(lambda x: x >= age_min[k] and x < age_max[k])]

#统计列中满足条件的数量
#sum()前面的返回值是一列值为False和True的列，将一列求和就得到满足条件的数量
(df_young[labels2[i]] >= optional_max[i]).sum() 

#创建多分类变量的哑变量
pd.get_dummies(df['所属职业分类'])

#替换某值
data["gender"] = data["gender"].map({"男":1, "女":0})  #map适用字典进行映射，将男替换成1，女替换成0

#创建日期
dates=pd.date_range('20231030',periods=6) #periods指定产生日期的天数，字符串指定起始日期，返回日期列表year-month-day

#按某列的值进行排序
df=df.sort_values(by='列名',ascending=False) #ascending排序方式，True升序，False降序
#按索引进行排序
df=df.sort_index(ascending=True,axis=0)#axis=0对行索引排序
```

## numpy库

```python
#产生一定范围的矩阵
#linspace能够精确控制终止值终值，而arange能够更直接地控制序列中值之间的增量
x=np.arange(-10,10,1) 
x=np.linspace(start=0,stop=10,num=5,dtype=int)
#产生一个全为0的矩阵
cell=np.zeros((length,width),int)

#将列表转化为矩阵类型
a=np.array('列表')
np.array(df,float)  #将DataFrame类型转化为numpy数组

#np.r_：是按行连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
#np.c_：是按列连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
#使用该函数时，两个连接的矩阵维数要相同
A = c_[zeros(4),diag([0.025,0.015,0.055,0.026])]

#numpy.diag(v,k)
#如果v是二维数组，返回k位置的对角线。
#如果v是一维数组，返回一个v作为k位置对角线的二维数组。
#k>0返回对角线上方，k<0返回对角线下方
np.diag([0.025,0.015,0.055,0.026])

#将numpy产生的矩阵重新塑造
array3 = np.arange(8).reshape(4, 2)
arr.reshape(m,-1) #改变维度为m行、d列 （-1表示列数自动计算，d= a*b /m ）
arr.reshape(-1,m) #改变维度为d行、m列 （-1表示行数自动计算，d= a*b /m ）

#求矩阵各个维度的最小/最大值
a.min()#全部元素中的最小值
a.min(axis=0)#每列的最小值
a.min(axis=1)#每行的最小值

#统计矩阵非零的个数
np.count_nonzero(a,axis=0) #a为数组名，axis用法与求矩阵最大最小值相同

#对矩阵进行排序  
np.argsort(x) #返回从小到大的索引值

#对两个矩阵进行连接
np.concatenate((np.eye(6), np.eye(6)),axis=1)#axis=1表示行拼接
res = np.hstack((arr1, arr2))      			 #np.hstack将参数元组的元素数组按水平方向进行叠加
```

##matplotlib库

## scipy库

```python
'''
fun: 目标函数: fun(x, *args) x 是形状为 (n,) 的一维数组 (n 是自变量的数量)

x0: 初值, 形状 (n，) 的 ndarray

method: 求解器的类型，如果未给出，有约束时则选择 'SLSQP'。

constraints: 求解器 SLSQP 的约束被定义为字典的列表，字典用到的字段主要有：

'type': str: 约束类型：“eq” 表示等式，“ineq” 表示不等式 (SLSQP 的不等式约束是 f ( x ) ≥ 0 f(x)\geq 0f(x)≥0 的形式)
'fun': 可调用的函数或匿名函数：定义约束的函数
bounds: 变量界限

'''
#求解非线性规划函数
res = scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

#显著性检验
#输出结果第一个值为pearsonr相关系数
#第二个为p-value，所以这里Guba列和Value值是显著相关的
stats.pearsonr(df['Guba'],df['Value'])

```













## 数学建模

```python
copy.deepcopy()   #深拷贝一个矩阵
random.gauss(miu,sigma)    #返回指定数学期望和标准差的高斯分布随机数
enumrate( 列表 )函数       #将列表组合为一个索引序列，同时列出数据下标和数据，一般用在 for 循环当中。
#numpy中比较两个矩阵是否相等
not(cell==cell_temp).all()     #.all()所有元素相同返回True   .any()任意一个元素相同返回True
#Matplotlib imshow() 方法
plt.imshow(cell)  #imshow() 函数可用于绘制矩阵、热力图、地图等。
#在pause函数运行前，图形将会更新并显示，在等待期间事件循环会一直运行，直到暂停时间interval秒后结束。
plt.pause(interval)  #interval表示秒数，类型为整数
```



##网络爬虫

**1.获取html网页**

**requests库**

主要用来发送http请求，会返回一个reponse对象，对象里包含了具体的响应信息,这些信息工具业务不同返回信息不同

```python
import requests
//发送get请求
response=requests.get(url)

response.json()       #如果它符合JSON格式，则可以使用json( )方法将其转换为字典类型，以方便解析
response.text         #获得返回对象的内容,返回的类型是str类型
response.content      #获取内容 byte类型
response.status_code  #获取状态码
response.headers      #获取响应头
response.encoding='utf-8'  #改变返回对象的编码方式
#设置headers 请求头
headers是解决requests请求反爬的方法之一，相当于我们进去这个网页的服务器本身，假装自己本身在爬取数据。
对反爬虫网页，可以设置一些headers信息，模拟成浏览器取访问网站 

```



**urllib.request库**

```python
# 导入urllib库的urlopen函数
from urllib.request import urlopen 
# 发出请求，获取html
html = urlopen("https://www.baidu.com/")
# 获取的html内容是字节，将其转化为字符串
html_text = bytes.decode(html.read())
# 打印html内容
print(html_text)
```

**2.解析html网页**

**BeautifulSoup库**

```python
soup = BeautifulSoup(html,'lxml')  #创建对象

soup.prettify()  #美化代码
#访问子节点和子孙节点
soup.body.h1    			# 选中body 标签下的h1，这个h1 标签是body标签的子节点
soup.div.find_all(‘img’)	#会找到所有div里面的img标签。

```



```python
'''
find()方法先找到对应的标签，{}里面是对应标签里面的键值对(属性)，也是用来查找用的,只返回单个元素
find_all()返回多个元素
get_text() 获取文本内容
string   获得文字信息，和get_text()方法类似

'''
#.children
children = soup.find('div',{'div',{'id':'u1'}}).children   #只返回单行标签内容
#.descendants
descendants=soup.find('div',{'id':'u1'}).descendants #会把标签中的文本内容单独返回(打印的时候可以发现区别)
#利用for循环遍历返回的对象
for child in children:
    print(child,child.get_text())
   
```



###urllib库

request模块：

parse模块：工具模块

error模块：异常处理模块   URLError

```python
#建立Request对象发起请求，可以添加headers模拟浏览器访问
from urllib import request,parse,error                                    
url='http://www.httpbin.org/post' 
#用来将字符串转化为字节流类型  发起请求时，方法中的data参数需要是字节流类型的 
#parse模块中的urlencode方法用来将字典转化为字符串
data=bytes(parse.urlencode({'name':'germey'}),encoding='utf-8')         
headers={                                                                 
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ',            
    'Host' : 'www.httpbin.org',                                           
    'Content-Type':'application/x-www-form-urlencoded'                    
}                                                                         
try:
    #不同于一般urlopen方法直接发起请求，这里先利用Request方法创建一个Request对象，然后利用这个对象发起请求
    #这种方式可以加入headers信息，添加User-Agent，用来模拟浏览器发起请求
    req=request.Request(url,data=data,headers=headers,method='POST')      
    response=request.urlopen(req)
	#处理可能发生的异常	
except error.URLError as e:
     print(e.reason)
else:                                                                     
     print(response.read().decode('utf-8'))                               
```

为了实现更高级的功能，需要深入一层的配置，原始的urlopen(也是个Opener)方法已经不适用了，需要构建新的Opener对象来发起请求.。**利用 Handler类来构建Opener类**

####HTTPBasicAuthHandler类
用来处理链接的认证问题

```python
#爬取有认证问题的网站
from urllib.request import HTTPPasswordMgrWithDefaultRealm,HTTPBasicAuthHandler,build_opener
from urllib import request
from urllib import error
#取消服务器证书验证功能，python默认开启服务器证书验证功能
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
url='http://ssr3.scrape.center/'
username='admin'
password='admin'
p=HTTPPasswordMgrWithDefaultRealm()
#将用户名和密码填入对象
p.add_password(None,url,username,password)
auth_handler=HTTPBasicAuthHandler(p)
#新的Opener对象
opener=build_opener(auth_handler)
try:
    result=opener.open(url)
    html=result.read().decode('utf-8')
    print(html)
except error.URLError as e:
    print(e.reason)
```



####HTTPCookieProcessor类
用于处理Cookie

```python
import urllib.request
import http.cookiejar
from urllib import error
import socket
filename='LWPcookie.txt'
#将Mazilla格式的cookie保存到Mozillacookie.txt文件中
#cookie=http.cookiejar.MozillaCookieJar(filename)
#将LWP格式的cookie保存到文件中
cookie=http.cookiejar.LWPCookieJar(filename)
cookie.save(ignore_discard=True,ignore_expires=True)
#用来处理Cookie方法,获得浏览器的cookie
handle=urllib.request.HTTPCookieProcessor(cookie)
opener=urllib.request.build_opener(handle)
url='http://www.baidu.com/404'
try:
    response = opener.open(url,timeout=0.01)
#先捕捉子类的错误,再捕捉父类的错误
#HTTPError是URLError的子类
except error.HTTPError as e:
    print(e.reason,e.code,e.headers)
except error.URLError as e:
    print(type(e.reason))
    #reson返回的不一定是字符串，利用isinstance方法判断返回结果是否为socket.timeout来判断是否超时
    if isinstance(e.reason,socket.timeout):
        print('TIME OUT')
else:
    print('Request successfully')
```



####robotparser模块
可以用于分析robot.txt文件判断网页是否可以抓取

```python
from urllib.robotparser import RobotFileParser
url='https://www.baidu.com/robots.txt'
#创建一个RobotFileParser对象
rp=RobotFileParser(url)
#执行read方法对文件进行分析，没有返回值，但如果不调用，接下来的判断都是False
rp.read()
#判断网页对于某个Uer-agent是否可以抓取
print(rp.can_fetch('Baiduspider','https://www.baidu.com/homepage'))
print(rp.can_fetch('Googlebot','https://www.baidu.com/homepage'))
```



###requests模块

```python
import requests
import re
#抓取网页
url1='https://ssr1.scrape.center/'
#将ssl证书验证关闭
request=requests.get(url1,verify=False)
#编写正则表达式
pattern=re.compile('<h2.*?>(.*?)</h2>',re.S)
title=re.findall(pattern,request.text)
print(title)
```

```python
import requests
import re
#抓取二进制数据
url2='https://scrape.center/favicon.ico'
r=requests.get(url2)
# print(r.text)
# print(r.content) 
with open('favicon.ico','wb') as f:
    #r.content获得的是响应的byte类型
    f.write(r.content)
```

# 网络



**访问网站的过程：**

域名解析 --> 发起TCP的3次握手 --> 建立TCP连接后发起http请求 --> 服务器响应http请求，浏览器得到html代码 --> 浏览器解析html代码，并请求html代码中的资源（如js、css、图片等） --> 浏览器对页面进行渲染呈现给用户



**DNS** 域名系统(Domain Name System)    将域名和IP地址相互映射的一个分布式数据库



**TCP**是一款面向连接的协议    **UDP**就被称为是一种“无连接”的协议    (**属于传输层**)   都是构建在 IP 协议之上。

当发起一个 TCP 连接时, 客户端首先自己先随机挑选一个没有被使用的端口作为服务器响应的接收端口, 比如 38672. 在一个 **TCP 的包**里, 无论是握手包还是后续的数据包, 包头部分最重要的两个字段, 一个就是**源端口(source port)**, 比如 38672; 另一个就是**目标端口(destination port)**, 比如 80, 或者 443. 而 TCP 包会作为 **IP 包**的数据包被打包到 IP 包里面, 也一个 IP 包里其实包含了 **IP + 端口.**

**端口**：一个数据包包括了文件，ip，和**端口号**，ip是为了服务器可以找到你的主机，端口号是你接受数据包的门户(多进程中的其中一个进程)， 而所谓的**端口监听**，是指主机网络进程接受到IP数据包后，察看其的目标端口是不是自己的端口号，如果是的话就接受该数据包进行处理。



超文本传输协议，即Hyper Text Transfer Protocol——**HTTP**，是一个为了使客户端和服务器顺利进行通讯而设计的协议。

http属于**应用层**，以下为常见的请求方法：

GET()——从指定的服务器中获取数据, 传输的参数直接暴露在URL上

POST()——提交数据给指定的服务器处理



**ping命令**：对一个网址发送测试数据包，看对方网址是否有响应并统计响应时间，以测试网络的连通性



**DHCP**（Dynamic Host Configuration Protocol，动态主机配置协议）通常被用于局域网环境，主要作用是集中的管理、分配IP地址，使client动态的获得IP地址、Gateway地址、DNS服务器地址等信息，并能够提升地址的使用率。



**交换机二层设备怎么能基于IP子网地址划分和基于高层应用或服务划分虚拟局域网呢？**

**基于 IP 子网地址划分**：这通常发生在三层交换机上，它们能够进行路由决策并基于 IP 地址信息来转发数据。在这种情况下，交换机可以根据源或目的 IP 地址来判断数据包应该属于哪个 VLAN。

**基于高层应用或服务划分**：这同样适用于三层交换机，或者更高层次的网络设备，如应用交付控制器。这些设备能够检查数据包中的应用层数据，如 HTTP、FTP 等协议类型，并根据这些信息来进行 VLAN 划分。



交换机只处理同一网段的数据帧转发，对于划分不同vlan的主机，就表示逻辑上在不同网段，需要三层交换机或者路由器来实现帧的转发

**ARP分组**封装在以太网帧中传播，在数据链路层就可以通过协议分割出源IP地址、源mac地址、目的IP地址这些字段，主机就可以根据这些字段来判断是否发给自己的ARP请求，如果是则发送ARP响应请求。

**主机发送ARP请求时，如何知道本网络路由器的IP地址从而获得路由器的mac地址？**主机通常是通过默认网关的设置来知道本网络路由器的IP地址的。默认网关就是一个网络设备（通常是路由器）的IP地址，它负责处理本地网络上的设备发送给其他网络的数据包。



突然有个想法，就是对于不同层级的数据，如果该数据是数据链路层（比如ARP请求）可以处理的，那么拆掉mac帧的首部和尾部就可以对数据部分，依照某种协议解释或提取数据部分中的字段，如果该数据数据链路层不能处理，那么就表示需要到更高层拆解，假设需要IP层（比如ICMP报文）来处理，那再数据链路层是的数据部分就是IP数据报，那么单靠数据链路层的协议是不能对数据进行解释的，因为IP数据报中也是有首部的，获得它的数据要进一步拆解。这样理解可以吗？



**为什么使用分类地址时，不存在最长前缀和匹配的情况？**最长前缀和匹配是因为路由器的转发表出现了多个网络前缀都匹配的情况，因为CIDR可以灵活的分配网络号，在某个网络号中继续划分更小的网络号，对于同一类的网络号（虽然网络号不同，但是属于一个公司的），可以对多个网络号进行**路由聚合**。但是分类地址的网络号固定，所以不会出现多个网络号聚合成一个网络号的情况，所以也就不会出现多个网络前缀都匹配的情况。



PING命令是应用层直接使用网络层ICMP的一个例子，没有通过运输层的TCP和UDP

TCP运输层可靠传输计时器：**超时重传计时器**（RTO，就是太久没有收到ACK然后重传数据）、**持续计时器**（就是在发送方接收到**零窗口**通知后，隔一个时间就发送个**零窗口探测报文**来检测是不是可以向接收方发数据了，避免了非零窗口通知丢失造成的死锁）、**时间等待计时器**(在TCP断开连接时，客户端和服务端都已经结束数据传送，客户端发送ACK表示收到服务端的FIN,但是防止ACK传输过程中丢失，所以要在等一会，服务端可能再一次传FIN。因为没有收到ACK服务端就不会断开连接)、**保活计时器**(防止客户端突然丢了，让服务端白白等待)

**对于积累确认，发送端该如何计算超时重传计时器的时间呢？**

**吞吐量？**吞吐量或网络吞吐量是指于一通信通道上单位时间能成功传递的平均资料量

#git命令



git pull

add . / add 文件名    //提交到缓冲区

git commit -m  "备注"    //提交到本地仓库

git push

git log   查看提交日志

git reflog	查看分支切换的日志

git config --list   //查看git配置

git config user.name 

git config user.email

git status  查看本地仓库的提交状态，红色表示修改未添加到暂存区，需要add

git branch   "分支名" 	创建分支

git rebase "分支名"。和git merge类似，这里是改变该分支的基准（根据原理理解）。

**利用SSH从本地创建文件提交至仓库的步骤：**

1. git init 创建本地仓库
2. git add .添加至暂存区
3. git commit -m " " 推送至本地仓库
4. ssh -T git@github.com连接github
5. 复制github仓库中的ssh地址。使用命令 git remote add origin  "ssh地址"    连接至仓库
6. git push -u origin master推送至远程仓库



**冲突处理：**

对于文本文件的冲突：Git把有冲突的段落标记出来了，上面是HEAD，也就是当前所在的分支，中间是分隔线，下面是另一个分支的内容。选择想要的删掉或留下，然后重新git add

**git正确开发流程**

正确的git开发流程

**第一步**

在github中创建一个新的仓库，这时候项目是空的，而且只有一个master分支

**第二步**

第一个开发人员进来了，他在本地创建一个develop分支，并且提交到远程

```mipsasm
git branch  develop
git push -u origin develop
```

现在线上就有两个分支master 和 develop 现在这两个分支里面都是空的

**第三步**

一、二步完成后，任何一个参与该项目的开发人员首先要做的就是从develop分支上切一个新分支进行功能开发

```xml
git checkout -b <本地分支名 feature/***> <origin/develop>
或者
git fetch origin 远程分支名:本地分支名
git branch --set-upstream-to=origin/远程分支名    本地分支名
```

然后进行开发，开发差不多，想提交一下

```sql
git status
git add
git commit
```

**第四步**

经过第三步，提交了几次后，感觉差不多了，就可以合并到develop分支

```cpp
git pull origin develop //先拉取develop中的代码，因为有可能别人已经往上提交过代码了
git checkout  develop//切到develop分支
git merge <feature/**>//合并feature/**中的代码到develop中
git push //提交到develop远程分支上
git branch -d feature/** //删除本地的分支
```

**第五步**

某一个开发人员想发布，但是其他人员还在进行开发，先不管别人，他先建立一个新的分支做发布准备

```xml
git checkout -b <本地分支名realse-0.1> <远程分支名develop>//注意这个realse-tagNo分支的功能是对发布的代码进行改善的地方
```

创建这个分支相当于测试环境修改，改好后就需要跟新master和develop,然后删除分支

```cpp
git checkout  master//切到master分支
git merge release-0.1//将release分支合到master上
git push//将合完的代码提交到远程master
git checkout develop//切到develop分支
git merge release-01//将release分支上的代码合到develop分支上
git push//合完的代码推送到远程的develop分支
git branch -d release-01//删除本地release分支
```

**第六步**

打tag追踪，这个过程不太了解

```bash
git tag -a -0.1 -m 'xxxxxx'
git push --tags
```

**Git有提供各种勾子（hook），即仓库有事件发生时触发执行的脚本
。可以配置一个勾子，在你push中央仓库的master分支时，自动构建好对外发布**

难道这就是传说中的自动化构建？？？

**第七步**

线上环境发现bug了

```cpp
git checkout -b hotfix/xxx master//从master分支上新建分支
```

然后开始改bug,改完后

```cpp
git checkout master//切回master分支
git merge hotfix/xxx//将改完bug后的代码合并到master
git push
```

改完bug的代码还要合到develop中

```bash
git checkout develop
git merge hotfic/xxx
git push
git branch -d hotfix/xxx
```



 	