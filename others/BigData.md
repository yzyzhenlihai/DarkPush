**RPC（Remote Procedure Call Protocol）远程过程调用协议：**

通俗的理解就是可以在本地程序中调用运行在另外一台服务器上的程序的功能方法。这种调用的过程跨越了物理服务器的限制，是在网络中完成的，在调用远端服务器上程序的过程中，本地程序等待返回调用结果，直到远端程序执行完毕，将结果进行返回到本地，最终完成一次完整的调用

RPC技术在架构设计上有四部分组成，分别是：**客户端、客户端存根、服务端、服务端存根**。

　　　　**客户端（client）**：服务调用发起方，也称为服务消费者。

　　　　**客户端存根(Client Stub)**：该程序运行在客户端所在的计算机机器上，主要用来存储要调用的服务器的地址，另外，该程序还负责将客户端请求远端服务器程序的数据信息打包成数据包，通过网络发送给服务端Stub程序；其次，还要接收服务端Stub程序发送的调用结果数据包，并解析返回给客户端。

　　　　**服务端(Server)**：远端的计算机机器上运行的程序，其中有客户端要调用的方法。

　　　　**服务端存根(Server Stub)**：接收客户Stub程序通过网络发送的请求消息数据包，并调用服务端中真正的程序功能方法，完成功能调用；其次，将服务端执行调用的结果进行数据处理打包发送给客户端Stub程序。



**WHL（Write-Ahead-Log）预写式日志：**

分布式环境必须考虑系统出错，Hbase采用HLog保证数据的一致性和实现回滚操作，用户更新的数据必须首先写入日志后，才能写入MemStore缓存，直到MemStore缓存内容对应的日志以及已经写入磁盘，该缓存的内容才能被刷写到磁盘。Region服务器每次启动都检查HLog文件，确认最近一次执行缓存刷新操作之后是否发生新的写入操作;如果发现更新，则先写入MemStore，再刷写到StoreFile，最后删除旧的Hlog文件，开始为用户提供服务。





**主从架构：**

HBase:  主节点 HMaster，从节点 RegionServer

MapReduce：主节点JobTracker（资源管理、任务调度、任务监控），从节点TaskTracker（任务执行）

Yarn：主节点ResourceManager（资源管理），ApplicationMaster（任务调度、任务监控），从节点NodeManager（原TaskTracker）

Spark：主节点Driver Program，从节点Worker Node，集群管理器Cluster Manager，由Driver创建的SparkContext负责和资源管理器进行资源的申请、任务的分配。集群管理器可以是Spark自带的资源管理器，也可以是YARN或Mesos等资源管理框架





**MapRedce中的Combiner与reduce的区别：**

Combiner类是用来优化MapReduce的，在MapReduce的Map环节，会产生大量的数据，Combiner的作用就是在map端先对这些数据进行简单的处理，减少传输到Reduce端的数据量，从而提高MapReduce的运行效率。Combiner并没有自己的基类，他是继承Reducer的，对外功能一样。他们的区别是，Combiner操作发生在Map端，在某些情况下Combiner的加入不会影响程序的运行结果，只会影响效率。

并不是所有求解都可以使用combiner function，对于求最大值或求和的功能，可以在中间结果的时候combine，并不会影响最终结果；但是对于求解平均值的功能，就不能对中间结果求解combine，会影响最终结果。





**yarn运行流程：**

1. client向RM提交应用程序，其中包括启动该应用的AM的必须信息，例如AM程序、启动AM的命令、用户程序等。
2. RM启动一个container用于运行AM。
3. 启动中的AM向RM注册自己，启动成功后与RM保持心跳。
4. AM向RM发送请求，申请相应数目的container。
5. RM返回AM的申请的containers信息。申请成功的container，由AM进行初始化。container的启动信息初始化后，AM与对应的NM通信，要求NM启动container。AM与NM保持心跳，从而对NM上运行的任务进行监控和管理。
6. container运行期间，AM对container进行监控。container通过RPC协议向对应的AM汇报自己的进度和状态等信息。
7. 应用运行期间，client直接与AM通信获取应用的状态、进度更新等信息。
8. 应用运行结束后，AM向RM注销自己，并允许属于它的container被收回。



**Hive执行过程：**

1. 编译器: SQL-->AST Tree-->QueryBlock-->OperatorTree。OperatorTree由很多逻辑操作符组成，可以在Map阶段和Reduce阶段完成某一特定操作。
2. ·优化器:对OperatorTree进行**逻辑优化**,来合并多余的操作符，以减少MapReduce任务数量以及Shuffle阶段的数据量。对优化后的OperatorTree进行遍历，生成需要执行的MapReduce任务。**物理优化**器生成最终的MapReduce任务执行计划。
3. 执行器:对最终的MapReduce任务进行执行。

