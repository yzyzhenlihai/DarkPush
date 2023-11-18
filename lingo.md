lingo结果解读

Global optimal solution found.
表示全局最优解找到
Objective value:
表示最优目标值
Total solver iterations:
表示用单纯行法进行了两次迭代
Variable
表示变量，运行结果中有两个变量为x1,x2
Value
给出最优解中个变量的值
Reduced Cost
与最优单纯形表中的检验数相差一个符号的数。
为了使某个变量在解中的数值增加一个单位，目标函数必须付出的代价（增加或减少Reduced Cost的值）
Slack or Surplus
表示接近等于的程度
在约束条件中是<=,叫做松弛变量
在约束条件中是>=,叫做过剩变量
在约束条件中是=,值为0,该约束为一个紧约束（或有效约束）
如果一个约束条件错误，作为一个不可行约束，Slack or Surplus为负数
Slack or Surplus表示的是：约束离相等还差多少
Dual Price
给出对偶价格的值
表示每增一个单位（约束右边的常数），目标值改变的数量（在最大化问题中目标函数是增加的，反之是减小的）
例如在本例中，c约束条件的Dual Price为1，表示2x1+x2<=600增加一个单位到2x1+x2<=601使目标值增加到-1(目标函数的Dual Price为-1），则Objective value就变为799
对偶价格也叫影子价格，这是由于他们表示可以用多大的价格去购买单位资源
