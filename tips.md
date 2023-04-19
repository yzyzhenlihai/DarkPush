##c++做题

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



10.   queue函数

    push() 在队尾插入一个元素

    pop() 删除队列第一个元素

    size() 返回队列中元素个数

    empty() 如果队列空则返回true

    front() 返回队列中的第一个元素

    back() 返回队列中最后一个元素



11. <algorithm>头文件 :  max()、min()、abs()、

12. typedef pair<int,int> PII   可以用来存储像坐标一样的数据类型

13. memset(dis,-1,sizeof(dis))  用来初始化数组     <cstring>头文件中

14. 在设置最大值时，可以使用    INF=0x3f3f3f3f  是一个很好的选择

    memset(a,INF,sizeof(a))。//通过这种方式，将数组初始化为无穷大



## java

**Object类**：每个类都直接或间接继承Object类，所有类型的基类

```java
String toString();//返回对象的字符串表示形式
int hashCode();//返回对象的散列码值
boolean equals();//判断两个对象是否相等
```



**String类**：创建对象后是个常量，内容和长度不可改变，如果对一个字符串进行修改需要创建一个新的字符串

```java
char charAt(int index);//返回位置上的字符
boolean equals(String string);//字符串比较
String toUpperCase();//将字符串大写
static String valueOf(int i);//将整型转换为字符串
char[] toCharArray(String str);//将字符串转化为字符数组
```

**StringBuffer类**：类似一个字符容器

```java
StringBuffer append(char c);//添加
StringBuffer insert(int index,String str);//插入
StringBuffer deleteCharAt(int index);//删除
```

**System类**:属性和方法都是静态的，所以可以直接用类名调用

```java
static void currentTimeMills();//返回以毫秒为单位的当前时间
//将源数组某个数开始复制到目标数组的某个位置，长度为length2
static void arraycopy(Object src,int srcPos,Object dest,int destPos,int length);
```

**Thread类**

```java
static void sleep(long millis);//使程序休眠millis毫秒，静态方法
```

 **Iterator接口**

```java
Iterator it=list.iterator();//获得对象的迭代器
void remove();//利用迭代器的方法删除集合中的某个数据
void hasNext();//判断集合中是否存在下一个元素
void Next();//获取迭代器指代的元素
```

**foreach循环**

不能修改集合中的元素

```java
for(Object obj : List);//Object 是集合中的元素类型
for(String str : strs);//访问字符串集合中的元素
```

**instanceof运算符**

用于强制转换前判断对象的真实类型，以免转换异常

**实现JDBC程序（连接数据库）**

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



**GUI(用户图形界面)**





## Python

str.isdigit()  //判断字符串是不是由数字组成

print(" ",%( ) )   //格式化输出

print(" ",end=' ')    //end=' '表示以' '结尾    ,不加end的话默认是换行

str()   //强转为字符串

find(子串，start，end)  //查找字符串中是否包含某个数，返回值为在字符串中的位置，没有返回-1

count(子串，start,end)  //查找子串出现的次数

rstrip()  //删除字符串右边的空白

in 是成员操作符，可以对序列（字符串、元组、列表、集合）做判断成员是否在其中

isdisjoint()  函数用于判断两个集合是否有相同的元素

空集合只能通过set()创建



**文件**

with open("文件名")  as  文件对象名 ：

​	contents=file.read()   //read() 方法读取文件的全部内容，并在文件末尾是返回一个空字符串，显示出来是个空行



**pyinstaller -F 测试.py**    cmd中打包py文件，生成可执行文件

-f 指定生成单独的exe文件(放在dis文件夹中)

-w 表示程序后台运行，没有窗口 （仅对 Windows 有效）

-c 表示提供程序视窗    （仅对 Windows 有效）

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

------



## 网络



**访问网站的过程：**

域名解析 --> 发起TCP的3次握手 --> 建立TCP连接后发起http请求 --> 服务器响应http请求，浏览器得到html代码 --> 浏览器解析html代码，并请求html代码中的资源（如js、css、图片等） --> 浏览器对页面进行渲染呈现给用户



**DNS** 域名系统(Domain Name System)    将域名和IP地址相互映射的一个分布式数据库

**TCP**是一款面向连接的协议    **UDP**就被称为是一种“无连接”的协议    (**属于传输层**)

超文本传输协议，即Hyper Text Transfer Protocol——**HTTP**，是一个为了使客户端和服务器顺利进行通讯而设计的协议。

http属于**应用层**，以下为常见的请求方法

GET()——从指定的服务器中获取数据, 传输的参数直接暴露在URL上

POST()——提交数据给指定的服务器处理



