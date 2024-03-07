## 异或运算

a^= b相当于a=a^b，将十进制数字转化为二进制进行运算，相同为0，相异为1，0和任何数异或运算都是原来的那个数。

可以用来判断数组中哪个数字只出现过一次 （通过将所有数与0进行异或运算）



## 快慢指针

1.  单链表中可任意用来寻找“**中点**”，快指针（fast）每一步走两个结点，慢指针（slow）每一步走一个结点。当快指针到达链表末尾时，慢指针应该指向链表最中间的结点.如果是**单数**恰好为中间，如果是**双数**则是中间的**第二个节点**
2. 在查找链表倒数第k个节点时，可以通过先让fast先走k步，再与slow同时运动，让两个指针==保持一个距离==，当fast到达结尾时，那么slow的位置就是倒数第k的位置。
3.  对于**环形链表**，把快慢指针想象成一个追及问题，当两个指针重合时，就代表链表中有**环**



## 记忆化搜索

## 前缀和



## 差分

**一维差分：**

首先给定一个原数组`a`：`a[1], a[2], a[3],,,,,, a[n];`

然后我们构造一个数组`b` ： `b[1], b[2], b[3],,,,,, b[i];`

使得 `a[i] = b[1] + b[2] + b[3] + ,,,,,, + b[i]`

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201215163431253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTYyOTI4NQ==,size_16,color_FFFFFF,t_70)

当想要求解原数组[ l , r ]区间同时加上c，可以通过求解差分数组b，使**b[l]+c，b[r+1]-c**

当除了b[1]以外的所有差分数组都为0时，表示原数组的值全都相等，等于b[1]

**二维差分：**

```c++
//将给定区间的数组都+c的这个操作封装成一个函数,b[]为差分数组，a[]为原数组，也是b[]的前缀和数组
void insert(int x1,int y1,int x2,int y2,int c){
    b[x1][y1]+=c;
    b[x2+1][y1]-=c;
    b[x1][y2+1]-=c;
    b[x2+1][y2+1]+=c;
}
b[i][j] = a[i][j] − a[i − 1][j] − a[i][j − 1] + a[i −1 ][j − 1]//差分数组求解，可以自己推导，很简单的
```



## 二分

```c++
/*
对于两个模板，首先寻找边界，最后如果check(mid)中边界的变化是r=mid，那么在计算mid的时候不用加1，如果check(mid)边界的变化是l=mid,那么在计算mid的时候需要加1
*/
//模板一，用于查找左边界，大于等于目标值的第一个数
int l=0,r=n-1;
while(l<r){
	int mid=l+r>>1;
	if(check(mid))r=mid;
	else l=mid+1;
} 
//模板二，用于查找右边界，小于等于目标值的最后一个数
int l=0,r=n-1;
while(l<r){
	int mid=l+r+1>>1;
	if(check(mid))l=mid;
	else r=mid-1;
} 

```

## 双指针

当遇到双指针**环的问题**的时候，可以**破环成链**，开两倍的数组存储两份相同的数据。

## 数论

**取模运算：**

1. (a + b) % p = (a % p + b % p) % p 
2. (a - b) % p = (a % p - b % p ) % p 
3. (a \* b) % p = (a % p \* b % p) % p 
4. 除法不可以直接同除，需要用到乘法逆元（还没学hh）

当运算多个数相乘再取模时，可以利用公式3进行运算，先对每个数取模后再计算乘积，这样乘积结果就不会溢出

**约数之和：**![image-20231203185920996](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20231203185920996.png)



其中pi都是质数，N可以拆成多个**质数幂**相乘，然后把约数之和的每一项拆开，就可以发现是每一种约数（就是从每个括号中选一个，并相乘，就可以得到一种情况的约数），所以约数的个数就是每个括号中**可以选择的数量**的乘积



**排序不等式：**

![img](file:///C:\Users\86159\AppData\Roaming\Tencent\Users\1826203343\QQ\WinTemp\RichOle\25F8E0YB@F8CL[B@K2A0FDR.png)

总的来说就是，升序和降序的序列，对应位相乘后求和能得到最小值，同为升序或者同为降序能得到和的最大值



## 递归

求解等比数列：这是一种常规的求解方式，或者利用公式法

```c++
/*
求解p^0-p^k-1的总和
需要分奇偶讨论
*/
int sum(int p,int k){
	if(k==1)return 1;
	if(k&1)return (sum(p,k-1)+qmi(p,k-1));//奇数
	if(!(k&1))return (1+qmi(p,k/2))*sum(p,k/2);//偶数
}
```



## 快速幂

**能够快速求出a^k mod p 的结果**，在计算mod时需要用到**第3条取模运算**的结论

正常的求幂的算法为O(N)，就是循环指数求解，但是循环次数过多，容易超时。快速幂的思想就是通过降幂的方式，减少指数，增大底数，从而起到减少循环次数的作用（本来的循环次数是n，通过多次指数除以2，底数平方，可以将循环次数降到很低很低），时间复杂度为O(logn)

```c++
/*	
	(a * b) % p = (a % p * b % p) % p  乘积的取模运算
	不断地将k除2降幂，将a底数平方
*/
ll qmi(ll a,ll k,ll p){
	ll res=1;
	while(k){
		//如果是奇数，就更新结果，将指数为1乘到结果上去 
		if(k & 1){
			res=res*a%p;
		}
		b>>=1;//指数除2
		a=(a*a)%p;//底数平方
	}
	return res;
}

```

## 分解质因数

试除法：时间复杂度为O(logn)~O(根号n)之间

试除法：每个合数都能分解为多个质因数相乘的形式（见数论的公式），合数N最多只有一个大于sqrt(N)质因子，证明：如果有两个大于sqrt(N)，那么相乘就会大于N，证毕。

```c++
for(int j=2;j*j<=a;j++){
/*
a%j==0的判断中j一定是质数，如果j是个合数，那么它能分解成多个质因子相乘的形式，这么多质因子也是a的质因子且都比j小（这些质因子是合数j分解的，所以肯定比j小）。
*/
			if(a%j==0){
				int s=0;
                //计算指数
				while(a%j==0){
					a=a/j;
					s++;
				}
				cout<<j<<" "<<s<<endl;
			}
		} 
		//a>1表示这是一个大于sqrt(N)的一个质数
		if(a>1){
			cout<<a<<" "<<1<<endl;
		}
```

## 筛素数

**埃氏筛：**时间复杂度O(nlogn*logn)，思想就是在筛选的时候，只把**质数的倍数**删掉，因为每个合数都能够由**多个质数**或者**两个质数**或者**一个质数一个合数**相乘，不像传统筛法，把每个数的倍数筛掉，这样会造成筛的重复，时间复杂度高

```c++
const int N=1e6+10;
int prime[N];//存质数
int st[N];//判断是否为质数
int cnt;//存质数的个数
for(int i=2;i<=n;i++){
        //没被筛掉
        if(!st[i]){
            prime[cnt++]=i;
           for(int j=i+i;j<=n;j+=i){
               st[j]=true;
           }
        }
        
    }
```

**线性筛：**每个合数只被最小质因子筛一次，感觉还不是很理解，先记一下结论

if(i%primes[j]==0) break; 

//当发现primes[j]是i最小质因子的时候,如果再继续进行的话，
//我们就把 prime[j+1]*i 这个数筛掉了,虽然这个数也是合数，
//但是我们筛掉它的时候并不是用它的最小质因数筛掉的，而是利用 prime[j+1] 和 i 把它删掉的
//这个数的最小质因数其实是prime[j],如果我们不在这里退出循环的话，我们会发现有些数是被重复删除了的。

```c++
 for(int i=2;i<=n;i++){
        //没被筛掉
        if(!st[i])prime[cnt++]=i;
        //线性筛，遍历每个质数
        for(int j=0;primes[j]*i<=n;j++){
            st[i*prime[j]]=true;
            if(i%prime[j]==0)break;//每个数只被最小质因数筛一次
        }
    }
    
```



## 并查集

一种树形的数据结构，近乎O(1)的时间复杂度。

又一次理解了并查集用来维护额外信息的作用，可以用来记录集合中的元素个数，也可以维护节点到根节点之间的距离，可能还有别的，然后在进行路径压缩的时候修改需要维护的额外信息。

主要构成 pre[]数组、find()、join()

1.可以将两个集合合并

2.询问两个元素是否在一个集合当中

```c++
//pre数组初始化
void Init(int n){
    //每个结点的祖先都是自己
    for(int i=0;i<n;i++){
        pre[i]=i;
    }
}
```



```c++
//pre[x]中存放的是x结点的父节点
//find()函数找到某个结点的根，结点的祖先
int find(int x)					//查找某个结点的父节点
{								
	while(pre[x] != x)			
		x = pre[x];				
	return x;					
}
```

**find()函数优化**：**路径压缩**。就是将所有结点的父节点都改为根结点，这样子查找某个结点的父节点只需要向上查找一次

```c++
int find(int x)     				//查找结点 x的根结点 
{
    if(pre[x]!=x)pre[x]=find(pre[x]);//如果没有找到根节点，就把每一个根节点都赋值给集合中的节点
    return pre[x];//最后返回的时候，每个节点的根节点都会改成同一个值（根节点）。
}
```

```c++
//将两个集合合并
void join(int x,int y)
{
    //找到各自的父节点
   	int fx=find(x);
    int fy=find(y);
    //将fy和fx其中任意一个作为根
    if(fy!=fx) 
        pre[fy]=fx;
}
```



## 数组模拟单链表

```c++
int e[N],ne[N],idx,head=-1;//head为链表的头指针
memset(ne,-1,sizeof ne);//需要对ne数组进行初始化，-1表示指向null
void insert(int x){
    e[idx]=x;//存的数据
    ne[idx]=head;//模拟指针，指向下一个数据
    head=idx;
    idx++;
}
//删除第k个数据,其实删除并没有真正的删除，而是通过修改ne数组（指针指向）来达到删除的目的
void del(int k){
    ne[k]=ne[ne[k]];
}
//遍历链表
void showList(){
    for(int i=head;i!=-1;i=ne[i]){
        //打印语句
    }
}
```

## 字符串哈希

![img](file:///E:\QQ\1826203343\Image\C2C\SBS`H6R2XYKUZ8QNI7$B8I3.png)

通过求解字符串前缀的哈希值的方式，可以比较字符串内任意字串的相等情况。首先需要把每个字符映射成数字，是什么无所谓（因为字符不好计算哈希值呀），然后类似于计算前缀和的方式，这里是计算h[i]表示前i个字符的哈希值。然后把要计算的每个前缀字符串看作是一个P进制的数（用于求解哈希值的），然后最终结果要映射的0~Q^n-1^的范围内，也就是要mod Q。对于P的经验值为131或者13331，Q一般是2^64^（此算法我们默认哈希不会产生冲突，不像前面的数值哈希会产生冲突）

通过求出的字符串前缀哈希值，可以用于判断字符串中子串的情况，99.99%的情况不会产生冲突（意思是说子串的哈希值不会冲突）。

求解l~r范围内的子串的哈希值，由于每一位都是有权重的，不可以直接通过前缀哈希值相减获得，需要考虑到权值。具体可以自己推，挺简单的（具体见代码）。



```c++
//计算字符串前缀哈希值
//应为求出的哈希值最终结果要映射的0~Q^n-1^的范围内，也就是要mod Q，这里用了一个巧妙的方法，就是用
//unsigned long long（64位）来存储哈希值，溢出后相当于mod 2^64
typedef unsigned long long ULL;
ULL h[N],p[N];//h[N]存字符串前缀哈希值,p[N]用于存P的某次方
const int P=131;//把字符传看作P进制的数
//和前缀和计算类似，这里字符串的下标也从1开始
p[0]=1;
for(int i=1;i<=n;i++){
    h[i]=h[i-1]*P+str[i];
    p[i]=p[i-1]*P;
}
//获得某字符串的哈希值
ULL get(int l,int r){
    return h[r]-h[l-1]*p[r-l+1];
}
```



##单调队列

单调队列一般适用于求解一段区间内的最大或最小值

```c++
//模拟队列
int q[N];
int hh=0,tt=-1;
q[++tt]=x;//入队
hh++;//出队
q[hh];//访问队头元素
hh<=tt;//表示队列不为空
//模拟栈
int stk[N];
int tt=-1;
stk[++tt]=x;//压栈
tt--;//出栈
stk[tt];//访问栈顶
tt>=0;//表示栈不为空
```

## Trie（字典树）

[Trie]([深度解析Trie（字典树）-CSDN博客](https://blog.csdn.net/raelum/article/details/128885107))，又称字典树或前缀树，常用来存储和查询字符串。假定接下来提到的字符串均由小写字母构成，那么Trie将是一棵 26 叉树。   

Trie存二进制数据或存字符串（以下代码为存二进制数）

```c++
ll a[N];//原始数据
ll son[M][2];//存Trie树
int idx;//Trie树结点下标
//插入Trie树 
//插入的步骤一般不会有变化，就是可能会增加cnt数组来维护字典树的额外信息，比如说经过某个结点的数的个数或者某个数出现的次数。
void insert(int x){
	int p=0;//树指针 
	for(int i=30;i>=0;i--){
		int t=(x >> i) & 1;//获得最高位
		if(!son[p][t]) son[p][t]=++idx;//son[p][t]=0表示根结点没有值为“t”的子结点
		p=son[p][t];
	}
}
//查询某个数
//查询就是遍历已经存在的字典树，完成相应的处理，比如说找到最大异或对，或者判断字符串是否已经出现过
ll query(int x){
	int p=0;
	ll res=0;
	for(int i=30;i>=0;i--){
		int t=(x>>i)&1;
		if(son[p][1-t]){
			res+=(1<<i);
			p=son[p][1-t];
		}else{
			p=son[p][t];
		}
	}
	return res;
}
```

## 邻接表



```c++
int h[N],e[M],w[M],ne[M],idx;//idx为边的序号
void add(int a, int b, int c)   // 添加有向边 u->v, 权重为weight
{
     e[idx] = b,w[idx] = c,ne[idx] = h[a],h[a] = idx ++;
    
}

```





## BFS

边权为1的求最短路问题可以用bfs做，因为是一层一层搜的，所以能找到最短路（第一次访问到的就是最短的）

```c++
const int N=110;
int g[N][N];//存图 
int d[N][N];//存每个点到源点的距离 
int n,m;
typedef pair<int,int> PII;
queue<PII> que;
int bfs(){
	int dx[]={-1,1,0,0},dy[]={0,0,-1,1};//上下左右
	que.push(make_pair(0,0));
	memset(d,-1,sizeof d);//初始化距离为-1，表示未经过
	d[0][0]=0;// 初始点为0
	while(!que.empty()){
		PII tmp=que.front();//取出队列中的第一个元素 
		que.pop();
		for(int i=0;i<4;i++){
			int x=tmp.first+dx[i],y=tmp.second+dy[i];
			if(x>=0 && x<n && y>=0 && y<m && d[x][y]==-1 && g[x][y]==0){
				d[x][y]=d[tmp.first][tmp.second]+1;
				que.push(make_pair(x,y));
			}
		}
	}
	return d[n-1][m-1];
	
}
```

**多源bfs**

对于多起点的bfs，可以在一开始将所有起点都加入队列，然后正常进行bfs，就可以得出所有点到起点的最近距离。

这也类似于**多起点的最短路问题**，如果边权不为1，需要利用Dijkstra来做，可以通过构建虚拟源点的方式求解，具体见**最短路笔记**



## DFS

强调顺序，然后再dfs完回溯后需要**恢复现场**，也就是把有一些标记的点重新清除标记。

```c++
//全排列代码
//t表示填充第t个位置
void dfs(int t){
	
	for(int i=1;i<=n;i++){
		if(!st[i]){
			st[i]=true;
			a[t]=i;//在第t位上填上数 
			dfs(t+1);
			//恢复现场
			st[i]=false;

		}
	}
}
```

## 拓扑序列

有向无环图一定存在拓扑序列，通过入度为0来判断该点是否可以加入队列。



## 最短路

**反向建图：**对于单源求最短路，可以直接使用Dijkstra或者spfa求解即可，对于**多起点单终点**的最短路问题，要计算每个起点到终点的最短距离，可以通过**反向建图**的方式，把起点和终点调换，这样就可以将问题转换为一个起点。

**虚拟源点：**通过建立虚拟源点也可以将多起点转换为从虚拟源点出发的新图。

### Dijkstra

适用于边权都为正数

**朴素版Dijkstra算法**

每次找到一个最小值，都会默认到该点的距离已经被更新至最小，所以用st数组进行标记，这也是Dijkstra算法的特点，利用贪心的思想，每次找到最短距离，就将这个点确定下来，不再更新。

```c++
const int N=510;
int g[N][N];
bool st[N];
int dist[N];//记录距离 
int n,m;//n为点的个数，m为边的个数
int dijkstra(){
	
	//初始化距离为最大值 
	memset(dist,0x3f,sizeof dist); 
	dist[1]=0;
	//找到距离的最小值
	for(int i=0;i<n;i++){
		int t=-1;
		for(int j=1;j<=n;j++){
			if(!st[i] && (t==-1 || dist[t]>dist[j]))t=j;//找到距离的最短点 
		}
		st[t]=true; 
		//更新距离最短点到其他点的最短距离
		for(int j=1;j<=n;j++){
			dist[j]=min(dist[j],dist[t]+g[t][j]);
		}
	} 
    //如果是最大值则表示无法到达
	if(dist[n]==0x3f3f3f3f)return -1;
	return dist[n];
}


```

**堆优化版Dijkstra算法**

堆优化版本在查找所有距离中的最小值时，使用的是堆，可以降低时间的复杂度。

```c++
const int N=2e5;
int n,m;
int h[N],e[N],ne[N],w[N],idx;
int dist[N];
bool st[N];
//稀疏图用邻接表 
void add(int a,int b,int c){
	e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
}
int dijkstra(){
	memset(dist,0x3f,sizeof dist);
	dist[1]=0;
    //用STL自带的优先队列存距离的值
	priority_queue<PII,vector<PII>,greater<PII> > heap;//优先队列-小根堆 
	heap.push(make_pair(0,1));//将第一个点加入堆
	while(!heap.empty()){
		PII t=heap.top();//获得距离中的最小值 
		heap.pop();
		int ver=t.second,distance=t.first;
		if(st[ver])continue;
		st[ver]=true;
        //跟新最短距离
		for(int i=h[ver];i!=-1;i=ne[i]){
			int j=e[i];
			if(dist[j]>distance+w[i]){
				dist[j]=distance+w[i];
				heap.push(make_pair(dist[j],j));//将新更新的距离加入堆 
			}
		}
	} 
	if(dist[n]==0x3f3f3f3f) return -1;
	return dist[n];
}
```

### SPFA

SPFA可以用于有**负权边**的单源最短路问题，时间复杂度为O(nm)，如果有**负权回路**的情况，则不能使用SPFA算法，否则会出现死循环。

因为SPFA是从源点开始，对于有发生松弛的点（就是最短距离发生改变），就会将其加入队列，算法结束的条件是队列为空，如果有负权回路，那么最短距离会永远有发生变化，那么队列中就会一直不为空。

SPFA判断是否存在负环，添加一个cnt数组统计两点之间边的个数，如果数量>=n，则表示一定存在负环。

```c++
int dist[N];
bool st[N];//st数组的用法与Dijkstra不同，这里是判断队列中是否会加入重复的点。
int que[N];
int spfa(){
    //初始化距离
    memset(dist,0x3f,sizeof dist);
    dist[1]=0;
    int hh=0,tt=0;
    que[0]=1;//加入队列
    st[1]=true;
    while(hh<=tt){
        int t=que[hh++];//取出
        st[t]=false;
        //更新距离
        for(int i=h[t];i!=-1;i=ne[i]){
            int j=e[i];
            if(dist[j]>dist[t]+w[i]){
                dist[j]=dist[t]+w[i];
                //如果没有在队列中就加入队列
                if(!st[j]){
                    que[++tt]=j;
                    st[j]=true;
                }
            }
        }
    }
    return dist[n];
    
}
```



### Floyd

可以处理负权

```c++
const int N=210;
int dist[N][N];
//floyd算法
void floyd(){
	for(int i=1;i<=n;i++){
		for(int j=1;j<=n;j++){
			for(int k=1;k<=n;k++){
				dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
			}
		}
	}
}
//dist数组初始化
for(int i=1;i<=n;i++){
    for(int j=1;j<=n;j++){
        if(i==j)dist[i][j]=0;//存在自环直接就删掉
        else dist[i][j]=INF;//初始化为最大值
    }
}
//因为可能存在负权边的情况，所以当两个点之间的距离无解时，可能距离并不是初始化的正无穷，而是会小一点点。
//所以用>INF/2来判断是否是无解。
if(dist[x][y]>INF/2)cout<<"impossible"<<endl; 
		else cout<<dist[x][y]<<endl;
```



## 最小生成树

### prim

时间复杂度为O(n^2)，可以使用堆优化，和Dijkstra一样用优先队列代替对，优化prim算法时间复杂度为O(mlogn)，适用于稀疏图，但是稀疏图的时候求最小生成树，Kruskal更加实用

```c++
int prim(){
	int res=0;
	memset(dist,0x3f,sizeof dist);
	//把第一个点加入集合 
	dist[1]=0;
	for(int i=0;i<n;i++){
		
		//找到到集合距离最短的 
		int t=-1;
		for(int j=1;j<=n;j++){
			if(!st[j] && (t==-1 || dist[t]>dist[j]))t=j;
		}
		if(dist[t]==0x3f3f3f3f)return dist[t];
		st[t]=true;
		res+=dist[t];
		//更新点到集合的距离
		for(int j=h[t];~j;j=ne[j]){
			if(dist[e[j]]>w[j]){
				dist[e[j]]=w[j];
			}
		}
	}
	return res;
}
```



###Kruskal

对边进行判断，首先对每一条边升序进行排序，然后遍历每一条边，加入生成树，如果最后生成树的边数=n-1，则表示生成树存在。

判断新加入的边是否构成环，可以通过**并查集**来维护一个集合。

```c++
const int N=1e5+10,M=2e5+10;

int n,m;
int res;
struct line{
	int u;//起点 
	int v;//终点 
	int w;//权重 
	//升序排序 
	bool operator < (const line& t){
		return this->w<t.w;
	}
	
}line[M];
int pre[N];
//并查集查找 
int find(int x){
	if(pre[x]!=x) pre[x]=find(pre[x]);
	return pre[x];
}
bool kruskal(){
	int cnt=0;
	//尝试加入每一条边
	for(int i=0;i<m;i++){
		int pa=find(line[i].u);
		int pb=find(line[i].v);
		if(pa!=pb){
			res+=line[i].w;
			pre[pa]=pb;
			cnt++;
		}
		
	} 
	if(cnt<n-1){
		return false;
	}
	return true;
}
```



## 动态规划

*例题一*：[最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

![image-20220822161641662](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20220822161641662.png)



*例题二:*   [买股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

动态转移方程：**dp[i]=min{dp[i-1],prices[i]}**,    dp[i]这个数组存着i位置前最小的价格



*例题三*：[分割数组以求得最大和]([1043. 分隔数组以得到最大和 - 力扣（LeetCode）](https://leetcode.cn/problems/partition-array-for-maximum-sum/))



==无后效性==：每一个子问题只求解一次，以后求解问题的过程不会修改以前求解的子问题的结果

## 背包问题（动态规划）

0/1背包： (物品只能选一次)

```c++
f[i][j] = max(f[i][j],f[i-1][j-v[i]]+w[i]);//二维
```

```c++
//滚动数组优化，当前行f[i]的状态只与上一行有关，所以可以用一维数组优化
//如果是从小到大,前一行的状态会被新一行的状态覆盖掉,这样使用前面已经求出来的状态就会出错
for(int j = m; j >= v[i]; j--)    //从大到小的重量
            f[j] = max(f[j], f[j - v[i]] + w[i]);//一维
```



完全背包:(物品可以选无数次)

```c++
f[i][j]=max(f[i][j],f[i][j-v[i]]+w[i])//二维转移方程
```

```c++
//从小到大可以满足一件物品不止选一次,物品不选则是前一行的状态，选该物品，则是同一行前面的状态再加上当前的价值
for(int j = v[i] ; j<=m ;j++)//注意了，这里的j是从小到大枚举，和01背包不一样
    {
            f[j] = max(f[j],f[j-v[i]]+w[i]);//一维
    }
```

分组背包(同组的物品只能选一个)：

和01背包类似，但是最外层的循环是组的数量，对每个组中的物品都讨论选或不选的价值，取最大值

需要三层循环：第一层i循环组的数量，第二层j循环背包的容量，第三层k逐个判断每组物品的价值取最大值

```c++
for(int i=1;i<=Max;i++){
		for(int j=m;j>=0;j--){
			for(int k=1;k<=s[i];k++){
				if(j>=w[index[i][k]]){//index存的是第i组第k个数在原始数组中的下标，从而获得该物品的大小和价值
                    dp[j]=max(dp[j],dp[j-w[index[i][k]]]+v[index[i][k]]);
                }
			}
		}
		
	}
```



## 获得环形链表相交的节点

1. 利用[快慢指针](##3.快慢指针)求解：获得fast和slow相遇的节点meetNode
2. 利用一个**结论**：两个节点，一个从头结点head出发，一个从meetNode节点出发，两个节点**速度相同**，最后会在环形链表的**相交节点处**（入环的第一个节点）相遇。

```c++
ListNode* fast=head,*slow=head;//定义快慢指针
 while(fast&&fast->next){
    fast=fast->next->next;
    slow=slow->next;
    if(slow==fast){
       ListNode* meetNode=fast;//获得快慢指针相遇的节点
       while(meetNode!=head){//利用结论获得相交的节点处（入环的第一个节点）
           meetNode=meetNode->next;
           head=head->next;
         }
        return meetNode;
    }
 }
```



*以下是手写证明：*

![img](file:///D:\QQ\1826203343\Image\C2C\E331194655F113A5D88003F3C6DB446C.jpg)

## 复制带随机指针的链表

[复制带随机指针的链表_leetcode](https://leetcode.cn/problems/copy-list-with-random-pointer/)

1. 先将新拷贝的结点链接在原来结点的**后边**

2. 通过原链表来改变拷贝链表的随机指针  (最关键的一步)

   ` copyNode->random=cur->random->next;//cur是原链表的结点，copyNode是拷贝的结点`

3. 将拷贝链表和原链表还原

  

## 串的模式匹配

1. BF算法(暴力算法)

2. [KMP算法](https://blog.csdn.net/qq_37969433/article/details/82947411)

   ​	求next的算法(关键代码):

```c++
	//求next的过程 默认字符串的下标是从1开始的
	for(int i=2,j=0;i<=n;i++){
		while(j&&p[i]!=p[j+1])j=ne[j];
		if(p[i]==p[j+1])j++;
		ne[i]=j;
	}
	//kmp匹配
	for(int i=1,j=0;i<=m;i++){
		while(j && s[i]!=p[j+1])j=ne[j];
		if(s[i]==p[j+1])j++;
		if(j==n){
			printf("%d",i-n+1);
			j=ne[j];
		}
	} 
```



## Huffman算法(哈夫曼算法)

通过生成**哈夫曼树**(最优二叉树)来进行**哈夫曼编码**

1. 将有权值的叶子结点按照从小到大的顺序排列

2. 取两个最小权值得结点作为新结点得左右孩子，小的为左孩子，大的为右孩子

3. 将新结点加入有序排列，继续重复步骤二

   

Huffman编码的**平均编码长度**计算：

先通过哈夫曼算法构造出最优二叉树后，判断每一个字符的编码长度，最后将**编码长度**乘以每个字符出现的**概率**求和

​	

## 高精度加法

1. 因为输入的数大于long long了，所以就用string先存着；

2. 将string里存的数逆序存入数字数组，这样模拟手工从右往左计算过程。

3. 循环（长的那个数组有多少个数，就循环多少次），两数相加，如果数>10，那就保留各位，十位加到下一个数中。

4. 因为数逆序存入所以要逆序输出。

   ```c++
   string s1,s2;
   int a[250],b[250],c[500];
    
   int main()
   {
   	cin>>s1>>s2;
   	
   	for(int i=0;i<s1.size();i++)  //将s1字符串逆序存入数组a,将s2字符串逆序存入数组b
   	{
   		a[s1.size()-i-1]=s1[i]-'0';  
   	}
   	
   	for(int i=0;i<s2.size();i++)
   	{
   		b[s2.size()-i-1]=s2[i]-'0';
   	}
   	
   	int len=s1.size();
   	if(s2.size()>len)
   	{
   		len=s2.size();
   	}
   	
   	for(int i=0;i<len;i++)
   	{
   		c[i]=a[i]+b[i];
   	}
   	
   	for(int i=0;i<len;i++)   //对进位进行处理
   	{
   		if(c[i]>=10) c[i+1]=c[i+1]+c[i]/10;
   		c[i]=c[i]%10;
   	}
   	
   	if(c[len]!=0) len++;    .//如果最高位有进位，那么c[len]还会有值
   	
   	for(int i=len-1;i>=0;i--)
   	{
   		cout<<c[i];
   	}
   	cout<<endl;
   	return 0;
   }
   ```

   

## st表(Sparse Table，**稀疏表**）

st表是一种数据结构，主要用于解决RMQ（区间最大值或最小值） 例如：给你一个数列，求解在一个范围内的数值的最大值或最小值。

主要利用了**倍增**的思想和**动态规划**的思想。



1. 动态规划的**预处理**（以2为倍数增加长度）

   

   ![img](https://img-blog.csdnimg.cn/img_convert/9767786ef22208934da9fb4faa16cb96.png)

   ```c++
   //由上图得，将要求得一个区间分为两个区间
   //f[j][i]存的是从 j 到 j+2^i-1 范围内的 最大值 ，中间包含2^i个数
   
   for(int i=1;i<=log2(n);i++){    // <<是移位运算符，1<<i相当于2^i
   		for(int j=1;j+(1<<i)-1<=n;j++){
   			f[j][i]=max(f[j][i-1],f[j+(1<<(i-1))][i-1]); 
               												}
   	}
   ```

3. 进行区间查询

   

   ![img](https://img-blog.csdnimg.cn/img_convert/27915cff73812cc0bd6a87333b08b2e7.png)

   找到一个值s，使得l+2 ^ s-1，尽可能接近r，r-2 ^ s+1尽可能接近l，两个区间的长度都是2 ^ s

   最后比较两个区间的最大值，取较大的那个

   ```c++
   max(f[l][s],f[r-(1<<s)+1][s]); 
   ```



## 归并排序  稳定排序

时间复杂度O(nlogn)    空间复杂度O(n)

就是给定两个有序数组，将两个数组合并在一起升序。

定义一个更大的数组，给定两个指针分别指向两个数组，每次取较小值放入新数组。

```c++
//1.分离函数
void mergesort(int x,int y)			//分离，x 和 y 分别代表要分离数列的开头和结尾
{
	if (x>=y) return;        //如果开头 ≥ 结尾，那么就说明数列分完了，分的只有一个数了，就返回
	int mid=(x+y)/2;            //将中间数求出来，用中间数把数列分成两段
	mergesort(x,mid);			//左右两端继续分离
	mergesort(mid+1,y);        
	merge(x,mid,y);        //分离玩之后就合并,升序排序，从最小段开始
}
```

```c++
//2.合并算法

void merge(int low,int mid,int high) //将两段的数据合并成一段，每一段数据都已经升序排序
{
	int i=low,j=mid+1,k=low;
    //i、j 分别标记第一和第二个数列的当前位置，k 是标记当前要放到整体的哪一个位置
	while (i<=mid && j<=high)    //如果两个数列的数都没放完，循环
	{
		if (a[i]<a[j])
			b[k++]=a[i++];     //a[n](原始数组)和b[n](临时存数据的数组)为全局函数
		else
			b[k++]=a[j++];   //将a[i] 和 a[j] 中小的那个放入 b[k]，然后将相应的标记变量增加
	}        // b[k++]=a[i++] 和 b[k++]=a[j++] 是先赋值，再增加
	while (i<=mid)
		b[k++]=a[i++];
	while (j<=high)
		b[k++]=a[j++];    //当有一个数列放完了，就将另一个数列剩下的数按顺序放好
	for (int i=low;i<=high;i++)
		a[i]=b[i];                //将 b 数组里的东西放入 a 数组，因为 b 数组还可能要继续使用
}
```



## 快速排序   不稳定排序

最差时间复杂度O(n^2) 和冒泡排序一样，平均时间复杂度O(n*log2n)

递归算法：

```c++
void sort(int* a,int l,int r){
	if(l>r)return;
	int i=l,j=r;
	int std=a[l]; //最左端作为标准值  因为标准值是最左端的数，所以要先从右边开始找
	while(i!=j){
		while(a[j]>=std&&i<j){//从右往左 找到比标准小的数 
			j--;             
		}
	
		while(a[i]<=std&&i<j){//从左往右 找到比标准大的数 
			i++;
		}
		if(j>i){  //交换找到的两个值
			int t=a[i];
			a[i]=a[j];
			a[j]=t;
		}
	}
	//退出循环表示i==j,将标准值换到i=j的地方，继续递归运行
	a[l]=a[i];   
	a[i]=std;
	sort(a,l,i-1);
	sort(a,i+1,r);
}
```

## 快速选择查找 (基于快速排序)    

可以用来查找第k小(大)的数，在快速排序每一轮确定基准值位置的时候判断是否是要选择的数

平均时间复杂度为O(n)，最坏时间复杂度为O(n^2)

```c++
//快速选择排序算法
int quickChoose(int left,int right,int k){
	if(left>right)return -1;
	int i=left,j=right;
	int std=num[left];
	while(i!=j){
		while(i<j&&num[j]>=std)j--;
		while(i<j&&num[i]<=std)i++;
		if(i<j){
			int t=num[i];
			num[i]=num[j];
			num[j]=t;
		}
	}
	num[left]=num[j];
	num[j]=std;
	if(k-1>j)quickChoose(j+1,right,k);//表示要查找的数在基准值位置的右边
	if(k-1<j)quickChoose(left,j-1,k);//在基准值的左边
	if(k-1==j)return num[k-1];//返回第k小(大)的值
} 
```

## 桶排序

不基于比较的排序算法

通过统计值域内每个数据的个数，然后根据个数排序

```c++
int count[1002];//存放数的个数 这里数的值域是[0,1000]
void bucketSort(int n){
    //统计每个数据的个数
	for(int i=0;i<n;i++){
		count[num[i]]++;
	}
	int cnt=0;
    //值域为[0,1000]
	for(int i=0;i<=1000;i++){
		while(count[i]!=0){
			num[cnt++]=i;//将数填回原数组
			count[i]--;
		}
	}
}
```



##堆排序(升序)   不稳定排序

时间复杂度  O(nlogn)

空间复杂度 O(1)

```c++
//对某个根节点调整为大根堆
//从上往下进行调整
void adjust_down(int *a,int n,int i){//i是需要调整的根节点的下标
	int father=i;
	int child=i*2+1;
	while(child<n){
        //比较孩子结点的大小，选出较大的那个
		if((child+1)<=n-1&&a[child]<=a[child+1]) ++child;
        //交换父节点和孩子结点，并顺着孩子结点向下继续调整
		if(a[father]<a[child]){
			int temp;
			temp=a[child];
			a[child]=a[father];
			a[father]=temp;
			father=child;
			child=father*2+1;
		}else{
            //一旦不能继续调整就退出循环
			break;
		}
	}
	
}


void heap_sort(int *a,int n){
	//建立大根堆
    //(n-1)/2为从后往前，第一个有孩子的结点
	for(int i=(n-1)/2;i>=0;i--){
		adjust_down(a,n,i);
	} 
	//摘取大顶,与最后一个结点交换 
	for(int i=0;i<n-1;i++){
		int temp=a[0];
		a[0]=a[n-i-1];
		a[n-i-1]=temp;
		adjust_down(a,n-i-1,0);
	} 
}
```





## Kruskal算法--最小生成树



```c++
//并查集中的查找父节点函数
int find(int x){
	if(pre[x] == x) return x;
	return pre[x]=find(pre[x]);//路径压缩
} 
struct edge{
	int u;
	int v;
	int w;
};
edge a[M];//定义边的数组
//n是结点的个数，m是边的个数
int kruskal(int n,int m){
	int cnt=0;//统计加入生成树的边的条数
	for(int i=1;i<=m;i++){
		//找到边的两个端点的父节点，判断是否会形成环 
		int father1=find(a[i].u);//find()函数，找到该节点对应的父节点(和哪些点连通)  具体见并查集
		int father2=find(a[i].v);
        //如果两个父节点相同，表示加入这条边会在图中形成环
		if(father1==father2){
			continue;
		}else{
            //表示可以加入图    
			pre[father1]=father2;//将两个结点的父节点相连(并查集中的合并操作)
			ans+=a[i].w;//ans统计最小生成树的权值和
			cnt++;
			if(cnt==n-1)break;
		}
	}
    //输出最小生成树的长度和
	if(cnt==n-1){
		cout<<ans;
	}else{
		cout<<"orz";//表示不是连通图，不能构成生成树
	}
	
}
```

```c++
//测试函数
#include<iostream>
#include<algorithm>
using namespace std;
const int N=5100;
const int M=200050;
int pre[N];//并查集
int ans; //统计最小生成树的长度和
bool compareEdge(edge a1,edge a2){//伪函数 
	return a1.w<a2.w;//升序 
}
int main(){
	int n,m;
	cin>>n>>m;
	for(int i=1;i<=m;i++){
		cin>>e[i].u>>e[i].v>>e[i].w; 
	} 
	sort(e+1,e+m+1,compareEdge);//排序
	//初始化pre数组
	for(int i=1;i<=n;i++){
		pre[i]=i;
	} 
	kruscal(n,m);
	return 0;
}
```

## 红黑树

特征：

1.**根节点**是黑色的

2.红色结点和黑色结点**交替**

3.任意节点到叶子结点的路径包含**相同数目**的黑色结点  (红黑树中的叶子结点是指**最外部的空结点**)



## 多重邻接表(存储无向图)

![image-20230214104351092](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230214104351092.png)

## 十字链表(存储有向图)

![image-20230214104512747](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230214104512747.png)



## AOV网(解决拓扑排序问题)---活动赋予顶点

方式：每次选出入度为0的结点

用邻接矩阵  时间复杂度 O(o^2)

用邻接表    时间复杂度  O(n+e)

 

## 哈希表平均查找长度

**成功时：ASL=(所有元素查找成功的比较次数)/元素的个数**

哈希函数为 H(key)=key mod p   （p一般来说是一个质数，**除留余数法**）

**不成功时：ASL=(0到p-1向后查找直到遇到空的比较次数)/p**      (不是除以数组本身大小)



## 全排列问题

深度优先搜索思想DFS :

```c++
int arrange[N];//存全排列的数组 
int exist[N];//判断是否存在 
void createArrange(int k,int n){//k是数组的位置，n是数的个数
	if(k>n){
		showArrange(n);//如果k>n表示所有数组位置都已经赋值，一次全排列情况结束，打印数组
		return;
	}else{
		for(int i=1;i<=n;i++){
			if(!exist[i]){
				arrange[k]=i;//如果有没有被填入数组的数，就填入，然后标记已经选过
				exist[i]=1;
				createArrange(k+1,n);//进入递归
				exist[i]=0;//当递归退出表示一次全排列结束，会返回上一层递归的位置，执行下面的语句,将当前的数标为
                				//还没有选过
				arrange[k]=0;
			}
		}
	}
}
```

**递归情况如图：(以n=3为例)**

![image-20230316230903951](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230316230903951.png)



## 遗传算法

遗传算法解决最优化问题的搜索算法

1. 基因编码

   二进制编码   010010011011011110111110

   浮点是编码   1.2 –3.3 – 2.0 –5.4 – 2.7 – 4.3

2. 建立表现型到基因型的映射关系，就是将基因映射到一个区间范围内(有点像解码的过程)

3. 建立适应性函数，衡量物种是否能够存活的标准

4. 选择函数，物种繁殖的概率，一般适应度占比越大，被选择的概率就越大. 使用轮盘赌

5. 遗传变异

   交叉：  **二进制编码**随机交换同一位置的的编码，产生新的个体

   ​		     **浮点数编码**就产生介于父代和母代基因编码值之间的数

   基因突变 ：二进制编码将某些位的基因改变

   ​					浮点是编码就是增加或减少一个小随机数

  ```python
#解决函数求极值问题
# 遗传算法
import numpy as np

DNA_SIZE=8   #编码的位数
POP_SIZE=200 #种族的个数
CROSSVER_RATE=0.9   #交叉的概率
MUTATION_RATE=0.01   #变异的概率
N_GENERATIONS=5000  #迭代的次数
X_BOUND=[-3,3]  # x的取值范围
Y_BOUND=[-3,3]  # y的取值范围

def F(x,y):
    return (x+y)**2+np.sin(y)

# 得到最大适应度
def get_fitness(pop):
    x,y=translateDNA(pop)
    pred=F(x,y)#获得函数值
    return (pred-np.min(pred))+1e-3   #加上个较小值，防止适应度小于零
#将每条二进制编码的基因投影到定义域中
def translateDNA(pop):
    '''
    解码
    :param pop: 种群矩阵，一行表示一个二进制编码的个体（可能解），行数为种群中个体数目
    :return: 返回的x,y 是一个行 为种群大小 列为 1 的矩阵 每一个值代表[-3,3]上x,y的可能取值（十进制数）
    '''
    x_pop=pop[:,1::2]  # pop中的奇数列表示x   对每一个染色体，从奇数列位x，从1开始
    y_pop=pop[:,0::2]  # pop中的偶数列表示y
    #dot函数，获得两个数的乘积   arrange()生成范围内的数
    #-1表示逆序遍历列表
    x=x_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    y=y_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(Y_BOUND[1]-Y_BOUND[0])+Y_BOUND[0]
    return x,y

# population_matrix = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
# print(len(translateDNA(population_matrix)[0]))

# 交叉、变异
def crossover_and_mutation(pop,CROSSVER_RATE=0.8):
    #定义新的种族
    new_pop=[]
    #对每一挑染色体都进行交叉或变异
    for father in pop:    # 遍历种群中的每一个个体，将该个体作为父亲
        child=father      # 孩子先得到父亲的全部基因（代表一个个体的一个二进制0，1串）
        if np.random.rand() <CROSSVER_RATE:  #产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother=pop[np.random.randint(POP_SIZE)]  # 在种群中选择另一个个体作为母亲
            cross_points=np.random.randint(low=0,high=DNA_SIZE*2)  #随机产生交叉的点 0-15
            child[cross_points:]=mother[cross_points:]   #母亲交叉点往后全部给孩子
        mutation(child)
        new_pop.append(child)
    return new_pop

def mutation(child,MUTATION_RATE=0.1):
    if np.random.rand()<MUTATION_RATE:
        mutate_points=np.random.randint(0,DNA_SIZE*2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_points]=child[mutate_points]^1    # 和1异或  将变异点位置的二进制反转

def select(pop,fitness):   # 自然选择，优胜劣汰
    #在两百条基因中中随机选出200条，replace=true表示可以有重复，p是每个元素采样的概率，根据适应度所占总适应度得到比例
    #200个中随机一个染色体
    idx=np.random.choice(np.arange(POP_SIZE),size=POP_SIZE,replace=True,p=(fitness)/fitness.sum())
    return pop[idx]

def print_info(pop):
    fitness=get_fitness(pop)
    max_fitness_index=np.argmax(fitness)#返回列表中最大值得索引值
    # print('此时种群',pop)
    # print('max_fitness:',fitness[max_fitness_index])
    x,y=translateDNA(pop)
    # print('最优基因型：',pop[max_fitness_index])
    # print('(x,y):',x[max_fitness_index],y[max_fitness_index])
    print('max_fitness:%s,函数最大值:%s'%(fitness[max_fitness_index],F(x[max_fitness_index],y[max_fitness_index])))

if __name__=='__main__':
    #获得200条长度为16的二进制编码，x,y各占8位
    pop=np.random.randint(2,size=(POP_SIZE,DNA_SIZE*2))
    for i in range(N_GENERATIONS):
        #x,y=translateDNA(pop)
        pop=np.array(crossover_and_mutation(pop))  # 交叉变异获得新的种群
        fitness=get_fitness(pop)  # 得到适应度
        pop=select(pop,fitness)   # 优胜劣汰
        if(i%100==0):
            print('第%s次迭代:'%i)
            print_info(pop)
    print_info(pop)

  ```



## 数学建模算法

梯度下降算法

 决策树模型

梯度提升决策树算法GBDT

随机森林

集成学习思想

正则项函数

非线性多目标规划模型

强监督学习模型

Booting算法

多项式拟合

加法模型

###数据预处理

1. 数据清洗
   * 缺失值
     * 删除法（缺失的太多了）
     * 替换法（用均值或众数）
     * 插值法
       * 牛顿插值法
       * 三次样本插值法
     * 函数拟合（样本数较多）
   * 异常值
     * 正态分布3σ原则 	数值分布在（μ-3σ，μ+3σ）中的概率为99.73%
     * [箱型图法](https://zhuanlan.zhihu.com/p/628585725) 普遍适用
2. 数据变换
   1. 
      * 将不是正态部分->正态分布数据 （开放、取对数、Box-Cox变换）
      
      * 将平稳序列->平稳序列（时间序列
      
      * 对于**成分数据**，约束条件为定和约束，需要对有效数据进行转变，使之累加和为100%
      
      * 中心对数比变换（clr）
      
        ![image-20230822100442062](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230822100442062.png)
      
   2. 数据规范化
      *  最大最小值归一化
      * 零-均值规范化（原始值-均值）/标准差
      * 小数定标规范化
      
   3. 引入哑变量
   
      所有类别哑变量的回归系数，均表示该哑变量与参照相比之后对因变量的影响。
   
      * **对于无序多分类变量，引入模型时需要转化为哑变量**
      * **对于有序多分类变量，引入模型时需要酌情考虑**，不同的有序分类变量并非严格是等距等比的关系，可以转化为哑变量进行量化
      * **对于连续性变量，进行变量转化时可以考虑设定为哑变量**，比如年龄变化的效应是很微弱的，可以将年龄进行划分，并赋值1、2、3、4，但以上赋值方式是基于一个前提，即年龄与因变量之间存在着一定的**线性关系**。因此，当我们**无法确定**自变量和因变量之间的变化关系，将连续性自变量离散化时，可以考虑进行哑变量转换。离散化分段统计，提高数据区分度
3. 数据分析法
   * 回归分析
   * 插值与拟合（插值是每个点都在函数上，拟合是得到最为接近的具体表达式）
   * 数据降维
     * 主成成分分析（从多个主要成分中找出主要的，让每个成分之间线性无关）
     * 因子分析

### 评价指标

**方差膨胀因子（VIF）**：VIF值代表多重共线性严重程度，用于检验模型是否呈现共线性，即解释变量间存在高度相关的关系（VIF应小于10或者5，严格为5）若VIF出现inf，则说明VIF值无穷大，建议检查共线性，或者使用岭回归。

**P值**：如果该值小于0.05，则说明模型有效；反之则说明模型无效

**OR值（odds ratio）**： 又称 比值比、优势比

OR值大于1，表示该因素是危险因素

OR值小于1，表示该因素是保护因素。

**AUC**（Area Under Curve）被定义为ROC曲线下的面积，下方面积越大，预测效果越好

**R^2^拟合优度**：拟合优度用于评价线性拟合的结果，定义拟合优度**R²**=SSR/SST。拟合优度越接近1，拟合效果越好。

SSR回归平方和、SST总体平方和、SSE误差平方和

**离差平方和（Sum of Squares of Deviations）**是各项与平均项之差的平方的总和

### 正则化

机器学习中经常会在损失函数中加入正则项，称之为[正则化](https://zh.wikipedia.org/zh-cn/正则化_(数学))（Regularize），为了防止模型过拟合而加入额外信息的过程



###回归算法

1. **线性回归**

   * **利用梯度下降法求解**

     ​	y=wx+b   (w,b都是需要求解的变量)

     1. 定义损失函数
     2. 选择起始点 （假设令w=0，b=0）
     3. 计算损失函数的梯度  （分别计算关于w和b的偏导数） 
     4. 按照学习率前进   （按照学习率重新获得新的w和b的值）
     5. 求解当前y的均方误差和上一次的均方误差，如果足够接近则表示已经获得相对正确的值

     ![image-20230722215652804](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230722215652804.png)

     <img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230722215559468.png" alt="image-20230722215559468" style="zoom: 50%;" />

     ```python
     import numpy as np
     import pandas as pd
     from matplotlib import pyplot as plt
     #线性回归利用梯度下降求解
     path = 'D:\\学习\\数学建模\\LinearRegressionTest_data.txt'
     data = pd.read_csv(path, header=None)
     plt.scatter(data[:][0], data[:][1], marker='+')
     data = np.array(data)
     m = data.shape[0]  #获得数组的行数   shape[1]可以获得数组的列数
     theta = np.array([0, 0])    #初始化两个θ的值
     data = np.hstack([np.ones([m, 1]), data])   #将参数的元组水平方向叠加，相当于在所有数据前都加上了 1
     y = data[:,2]  #获得y
     data = data[:,:2]     #获得前面两列的数据
     
     #损失函数
     def cost_function(data, theta, y):
         cost = np.sum((data.dot(theta) - y) ** 2)    #将data和theta两个矩阵相乘
         return cost / (2 * m)
     
     #求得梯度
     def gradient(data, theta, y):
         grad = np.empty(len(theta))   #创建未初始化的数组
         grad[0] = np.sum(data.dot(theta) - y)
         for i in range(1, len(theta)):
             grad[i] = (data.dot(theta) - y).dot(data[:, i])
         return grad
     
     #梯度下降法
     def gradient_descent(data, theta, y, eta):
         while True:
             last_theta = theta
             grad = gradient(data, theta, y)
             theta = theta - eta * grad
             print(theta)
             if abs(cost_function(data, last_theta, y) - cost_function(data, theta, y)) < 1e-15:
                 break
         return theta
     
     
     res = gradient_descent(data, theta, y, 0.00001)
     X = np.arange(3, 25)
     Y = res[0] + res[1] * X
     plt.plot(X, Y, color='r')
     plt.show()
     ```

   * **最小二乘法**(OLS)

     本质上就是直接利用导数求极值的方式直接获得最大值或最小值。与梯度下降法的区别就是，梯度下降法通过每次选择梯度最大的方向前进，不断改变w和b的值，直到达到相对最优解（可能会陷入局部最优解）
     
   
2. **逐步回归**

   逐步回归主要解决的是**多变量共线性问题**，也就是不是线性无关的关系，它是基于变量解释性来进行特征提取的一种回归方法。

   本质上是多元线性回归，逐步回归是一种方法，能降低模型维度，得到满意模型性能

   数据要求：

   1. 需要至少2个自变量，且**自变量之间互相独立**
   2. 自变量与因变量之间存在线性关系，可以通过绘制散点图予以观察
   
   3. **因变量为为连续变量**，自变量为连续型变量或分类变量
   4. 数据具有**方差齐性、无异常值和正态分布**的特点检验方法
   
   5. **自变量间不存在多重共线性**
   
   
   
3. **曲线回归**(多项式回归)

6. **logistic回归分析**

   Logistic回归分析属于非线性回归，研究**因变量为二项分类或多项分类结果与某些影响因素之间关系的一种多重(多元)回归分析方法。**

   构建的**分类器**是一个线性的分类器，其决策边界是一条直线（因为我们只使用了输入特征 x xx 的一次项）。为了得到一个非线性的决策边界，我们可以尝试构建输入特征的**多项式项**来刻画非线性关系。

   

   **[二分类logistic回归分析](https://blog.csdn.net/m0_60862600/article/details/122985622)、和有序Logistic回归分析**

   数据要求:

   1. 要求自变量之间**无多重共线性**

   2. Y为定类数据，X可以是定量数据或定类数据

      

   在线性回归的基础上加上了Sigmoid函数，使之成为分类算法

   分类输出的是离散数据（预测是否通过），回归输出的是连续数据（预测分数）
   分类得到的是决策面，用于对数据集中的数据进行分类；回归得到的是最优拟合线，这个线条可以最好的接近数据的每个点；
   对模型评估指标不一样：如何评估监督分类，通常使用正确率作为指标；回归中通常使用决定系数R平方表示模型评估的好坏，R平方可以表示有多少百分比的y波动被回归线所描述。

   **[多分类Logistic回归分析]((https://blog.csdn.net/m0_60862600/article/details/122986644?ops_request_misc=%7B%22request%5Fid%22%3A%22169234171916800180682340%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=169234171916800180682340&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-122986644-null-null.142^v93^chatsearchT3_2&utm_term=多分类逻辑回归&spm=1018.2226.3001.4187))：**

   ![logistic回归流程](https://img-blog.csdnimg.cn/b599b8ca1871414e8a3625b8821ddce7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA6buR5rSe5ou_6ZOB,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

   

   **One-vs-all分类思想：**将每一个类别都单独训练二分类logistic回归模型，对于每个新的输入，使用训练出的分类器**分别计算“样本属于每个类别的概率”**，进而，选择概率值最高的那个类别作为该样本的预测类别。

7. **决策树回归**

   与分类树用基尼指数最小原则不同，对回归树用平方误差最小化准则

8. **随机森林回归**

9. **梯度提升树回归**

10. **保序回归**

11. **L1/2稀疏迭代回归**



### 分类模型

 #### 聚类分析

对样本进行分类称为**Q型聚类**分析，对指标进行分类称为**R型聚类**分析。

![image-20230825201945937](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230825201945937.png)

**聚类变量重要性：**在给定数据集上进行聚类时，我们可以尝试**不同的聚类数**，并计算每个聚类数下的**轮廓系数**（聚类结果的评价指标）。轮廓系数是一种衡量聚类结果**紧密度**和**分离度**的指标，取值范围在-1到1之间，数值越接近1表示聚类结果越好。

**簇内不相似度：**计算样本![i](https://latex.csdn.net/eq?i)到同簇其它样本的平均距离为𝑎(𝑖)，应尽可能小。

**簇间不相似度**：计算样本𝑖到其它簇![C_j](https://latex.csdn.net/eq?C_j)的所有样本的平均距离![b_{ij}](https://latex.csdn.net/eq?b_%7Bij%7D)，应尽可能大。

![img](https://img-blog.csdnimg.cn/3741245c6acc480e910259ea0f9c4dfa.png)



**适用条件：**层次聚类算法对时间和空间需求很大，所以层次适合于小型数据集的聚类

####决策树分类

<img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20240128154310075.png" alt="image-20240128154310075" style="zoom: 50%;" />

信息熵，衡量集合中的混乱程度，当分类完成后，可以根据集合中熵的值来判断分类的程度，H(X)越大表示集合中越混乱，分类效果越差。

如果是**回归问题**的话，就不同能过类别计算特征的信息熵，而是通过方差，如果基于一个特征分类完后，集合中的值方差很小，表示分类效果不错，值都比较相近。然后对训练模型进行预测的话，可以通过计算集合中的平均值来作为预测值

**ID3：**计算信息增益，适用于每个特征都是**定类变量**，能够计算该特种中每个类别的信息熵，然后加权得到基于该特征分类的信息熵，再计算信息增益。但是如果对于定序或定距或类别比较多的特征，考虑一种极端情况，每个样本是该特征中的一个种类，那么该特征的信息熵计算出来就是0（因为该特征中每个类别的信息熵都是0，都只有一个样本对于这个类别），这就是ID3的缺陷。

**C4.5：**计算信息增益率，解决ID3的问题，考虑某特征的自身熵值。

**CART :**使用Gini系数当作衡量标准，和熵值类似

![image-20240128160653715](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20240128160653715.png)



### 集成算法

#### Bagging模型

典型的是随机森林，很多个决策树并行放在一起，数据随机采样来训练模型，树之间不产生干扰。

####Boosting模型

提升算法

#### Stacking模型



### BP神经网络

神经网络输入层到隐含层的第一组权值如何确定？

### 预测与评价方法

  **[评价类问题](https://www.cnblogs.com/haohai9309/p/17204510.html)**对因素规范化前，先对指标（中间型、极小型）进行极大化处理，就是全部转化为极大型指标，在进行归一化或标准化。

| 极大型（效益型）指标 | 越大（多）越好   | 成绩、GDP增速、企业利润  |
| -------------------- | :--------------- | ------------------------ |
| 极小型（成本型）指标 | 越小（少）越好   | 费用、坏品率、污染程度   |
| 中间型指标           | 越接近某个值越好 | 水质量评估时的PH值       |
| 区间型指标           | 落在某个区间最好 | 体温、水中植物性营养物量 |



####**移动平均法(Moving Average)**

用于对**平稳序列**进行预测

* **SMA(简单移动平均)**

  **固定跨越期限**内求**平均值**作为预测值

* **WMA(加权移动平均)**

  加权移动平均给**固定跨越期限**内的每个变量值以不同的权重

* **EMA(指数移动平均）**

  ![image-20230720113034098](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230720113034098.png)

  ![image-20230720113055371](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230720113055371.png)

  随着权重因子 β 的增大（β增大，表示先前的指数移动平均占比增大），指数移动平均曲线逐渐变得更加平滑，但同时指数移动平均值的实时性（当前值所占的比重减小，对平均值的影响减弱）也随之变弱。

  ```python
  
  '''
  指数平均和引入偏差修正公式后的指数平均预测对比
  '''
  import matplotlib.pyplot as plt
  import numpy as np
  
  
  def main():
      beta = 0.9    #权重系数
      num_samples = 100   #样本数量
  
      # step 1 generate random seed
      np.random.seed(0)
      raw_tmp= np.random.randint(32, 38, size=num_samples)
      x_index = np.arange(num_samples)
      # raw_tmp = [35, 34, 37, 36, 35, 38, 37, 37, 39, 38, 37]  # temperature
      print(raw_tmp)
  
      # step 2 calculate ema result and do not use correction
      v_ema = []
      v_pre = 0
      for i, t in enumerate(raw_tmp):
          v_t = beta * v_pre + (1-beta) * t
          v_ema.append(v_t)    
          v_pre = v_t   #记录前一个指数移动平均数
      print("v_mea:", v_ema)
  
      # step 3 correct the ema results
      v_ema_corr = []
      for i, t in enumerate(v_ema):
          v_ema_corr.append(t/(1-np.power(beta, i+1)))
      print("v_ema_corr", v_ema_corr)
  
      # step 4 plot ema and correction ema reslut
      plt.plot(x_index, raw_tmp, label='raw_tmp')  # Plot some data on the (implicit) axes.
      plt.plot(x_index, v_ema, label='v_ema')  # etc.
      plt.plot(x_index, v_ema_corr, label='v_ema_corr')
      plt.xlabel('time')
      plt.ylabel('T')
      plt.title("exponential moving average")
      plt.legend()
      plt.savefig('./ema.png')
      plt.show()
  
  
  if __name__ == "__main__":
      main()
  ```

#### 灰色预测模型

基于时间序列的预测



#### 长短期记忆（Long Short Term Memory，LSTM）网络

基于时间序列的预测，可以记住早先时刻的信息，是一种特殊的RNN模型（RNN，循环神经网络）

模型的自变量为历史数据，因变量未来某时刻的数据。





#### 马尔科夫预测





#### 灰色关联分析

灰色关联分析的基本思想是**根据序列（比如时间序列）**曲线几何形状的相似程度来判断其联系是否紧密。曲线越接近，相应序列之间的关联度就越大，反之就越小。**用于确定各因素对其所在系统里的影响因素（系统作为参考数列）**或者**综合评价，进行优劣排名（标准值作为参考数列）**

**灰色关联度计算方式：**

![image-20230819164509046](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230819164509046.png)

**关联度计算：**每列因素的灰色关联度**求均值**即为某因素的关联度

**指标权重计算：**通过各指标的关联度计算各指标的权重

**评价对象得分计算：**归一化后的指标*权重并求和

####熵权法 

算法步骤：

1. **数据归一化**

   ![image-20230724180734479](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230724180734479.png)

2. **计算第j项指标下第i个方案的指标值比重**

   <img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230724181012250.png" alt="image-20230724181012250" style="zoom: 80%;" />

3. **计算第j项指标的信息熵**

   <img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230724181146911.png" alt="image-20230724181146911" style="zoom:80%;" />

4. **计算各指标的权重**

   <img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230724181245080.png" alt="image-20230724181245080" style="zoom:80%;" />

5. **计算评价对象的综合评价值**

   <img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230725155220781.png" alt="image-20230725155220781" style="zoom: 50%;" />

#### 模糊综合评价法

因素集（评价指标集）  U = {u1,u2,…,un}
评语集（评价的结果集） V = {v1,v2,…,vm}
权重集（指标的权重）  A = {a1,a2,…,an}

评判结果集						    B=[b1,b2,...bn]

1. **一级综合模糊评价**

   主要用于考核涉及指标较少

   **评价步骤：**

   1. 确定因素集（考核的指标）

   2. 确定评语集（比如好、较好、中等、较差等评价）

   3. 确定各因素的权重（确定权重的方式可以是熵权法、层次分析法）

   4. 确定模糊综合判断矩阵

      ![image-20230725164402034](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230725164402034.png)

      该矩阵是通过定性分析得出的，每个人对ui指标按照评语集V进行评价，并统计出每个评语所占的比重，获得Ri

   5. 模糊综合评判

      综合评判的结果为     B=A⋅R

      bi的含义是要评价的对象对于评语 i 的**隶属度**，选取隶属程度最大的就是评价对象的评价结果

2. **多层次模糊评价**

   主要用于考核涉及的指标复杂，具有多层次（某个指标可以分为更细的评价指标）

   评价步骤与一级综合模糊评价类似，先对一级指标中的二级指标进行评判，得到的B就是更高层次评价的R（得到的评判结果作为高层次评价的判断矩阵），然后对一级指标重复上述步骤进行评价。

   **例如如下情况**

   <img src="https://img-blog.csdnimg.cn/20210112165617998.png?#pic_center" alt="img" style="zoom:50%;" />
   
   
   
   

#### 数据包络DEA分析法

用于评价效率问题

#### Topsis综合评价方法



### 仿真方法

1. Monte Carlo Method（蒙特卡洛法）

   蒙特卡罗方法又称统计模拟法、**随机抽样技术**，是一种随机模拟方法，多用于求解复杂的多维积分问题

   每次输入都随机选择输入值。由于每个输入很多时候本身就是一个**估计区间**(可能是均匀分布，也可能是正态分布)，因此计算机模型会随机选取每个输入的该区间内的任意值，通过大量成千上万甚至百万次的模拟次数，最终得出一个累计概率分布图
   
   * 取舍采样（rejection sampling）
   * 马尔科夫链MC采样 

2. 元胞自动机



### 差异性分析

Pearson:相关系数：用于衡量两个连续变量之间的线性关系。具有计算简单、解释方便、可比性强等优点，但缺点是对异常值敏感，对非线性关系不敏感。
Spearman: 相关系数：用于衡量两个变量之间的单调关系。它具有不受异常值影响、不要求数据呈正态分布等优点，适用于非线性单调关系的情况，但是可能会忽略掉数据间的差异信息。

Kendall 相关系数：也用于衡量两个变量之间的单调关系。与 Spearman
相关系数相比，它更加稳健，能够有效处理小样本问题，但是计算复杂度较高。

切比雪夫相关系数：用于衡量两个变量之间的距离或差异。它具有不受数据分布和缩放影响的优点，但是对于极端异常值的情况，可能不够稳健。

Eta 相关系数：用于衡量两个分类变量之间的关系。由于是基于卡方检验的效果量，因此具有显著性水平的信息，但是只能处理两个变量之间的关系。

互信息：用于衡量两个变量之间的非线性关系。它比 Pearson
相关系数更加灵活，能够处理多元关系和噪声干扰的情况，但是计算复杂度较高，需要大量的样本数据支持。

综上所述，不同的相关系数适用于不同的数据类型和研究目的，选择合适的方法能够得到更准确的结果。

1. **卡方检验（**观察频率预期频率之间是否存在显著性差异）**类别变量对类别变量**

   **适用于不服从正态分布的数据，两组变量是无序的**

   <img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230726174029512.png" alt="image-20230726174029512" style="zoom:33%;" />

   fo为观测值频数，fe为期望值频数。当计算出的卡方值大于卡方临界值时，就可以拒绝零假设 

   **卡方临界值**通过**自由度**+**α**（显著性水平）查表获得（一般α=0.05）

   * 卡方拟合度检验（单因素卡方检验）

     针对一个类别变量

     <img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230726174618582.png" alt="image-20230726174618582" style="zoom: 67%;" />

   * 卡方独立性检验（二因素卡方检验）

     针对两个类别变量

     <img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230726174649658.png" alt="image-20230726174649658" style="zoom: 67%;" />

     ​												fr为行观测频数的和，fc为列观测频数的和，n为样本总量

2. **独立样本t检验**

   二分类变量对连续变量

3. **单因素方差分析**

   多分类变量对连续变量

4. **相关型分析**

   连续变量对连续变量

   * **Pearson相关系数**

     Pearson相关系数用于评估两组数据是否符合线性关系，不能用于符合曲线关系的数据

     这种统计方法本身不区分自变量和因变量，

   * **Spearman相关系数**
   
     均为有序分类变量
     
     spearman相关系数适用于不满足线性关系，且不满足正态分布的数据
     
     **可以**认为是**定距变量:**
     
     **Mantel-Haenszel 趋势检验**。该检验也被称为Mantel-Haenszel 卡方检验、Mantel-Haenszel 趋势卡方检验。该检验根据研究者对有序分类变量类别的赋值，判断两个有序分类变量之间的线性趋势。
     
     **不能**认为是定距变量:
     
     Spearman相关又称Spearman秩相关，用于检验至少有一个有序分类变量的关联强度和方向
     





### 敏感性分析

是指从定量分析的角度研究有关因素发生某种变化对**某一个**或**一组关键指标**影响程度的一种不确定分析技术。每个输入的灵敏度用某个数值表示即敏感性指数



