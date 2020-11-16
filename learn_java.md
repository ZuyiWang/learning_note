# Java 学习体系 #
## 基础部分 JavaSE ##
1. 面向对象的编程思想 ArrayBox LinkedBox
2. 集合 String
3. I/O 流 MVC 缓存 文件--数据库 事务
4. 反射注解 IOC
		
## Level One ##
1. 数据库	本质就是文件 基本使用--性能--索引--引擎--锁--事务
2. JDBC  	本质就是I/O 手动设计一个ORM --原理 MyBatis
3. WEB 	本质就是Socket IO String解析
	* 手动设计服务器   Tomcat 
	* Servlet JSP解析------>手动设计一个WEB框架 
	* Filter AJAX    

## Level Two ##
1. SSM		Spring SpringMVC MyBatis
2. SSH		Spring Struts    Hibernate
3. Linux Maven Git SVN
4. 分布式  大数据

------

# Overload --- 方法重载 #
**方法重载**
1. 概念
	+ 一个类中的一组方法，相同的方法名字，不同的参数列表，这样的一组方法构成了方法重载；
	+ 参数列表的不同体现在：参数的个数，参数的类型，参数的顺序；
2. 作用
	+ 为了让使用者便于记忆与调用，只需要记录一个名字，执行不同的操作；
3. 自己设计方法重载
	+ 调用方法的时候，首先通过方法名字定义方法；
	+ 如果方法名字有一致，可以通过参数的数据类型定位方法；
	+ 如果没有与传递参数类型一致的方法，可以找一个参数类型可以进行转化（自动）
4. JDK1.5版本之后，出现了一个新的写法
	+ int... x 动态参数列表，类型固定，个数可以动态 0--n都可以；
	+ x本质上就是一个数组，有length属性，有[index]；
	+ 动态参数列表的方法，不能与相同意义的**数组类型的方法**构成方法重载，本质是一样的；
	+ 动态参数列表的方法，可以不传参数，相当于0个，数组的方法必须传递参数；
	+ 动态参数列表在方法的参数中只能***出现一次***，且***必须放置在方法参数的末尾***；
# 类中的四个成员 #
1. **属性**
	+ 静态描述特征(存值)；
	+ ***权限修饰符 [特征修饰符] 属性类型 属性名字 [= 值]；***
2. **方法**
	+ 动态描述行为(做事情)
	+ ***权限修饰符 [特征修饰符] 返回值类型 方法名字 ([参数列表]) [抛出异常] [{方法体}]；***
	+ 最主要的是方法设计的参数传递与返回值问题 传递 调用 内存 执行
3. **构造方法**
	+ 创建当前类对象(做事情，唯一的事情)
	+ 没有返回类型，但是又返回值；支持方法重载；
	+ ***权限修饰符 与类名相同的方法名字 ([参数列表]) [抛出异常] {方法体}；***
4. **程序块**
	+ 一个特殊的方法(没名，做事情，不用我们调用，构建对象之前自动调用，多个块按编写顺序执行)
	+ ***{方法体}***
5. **this关键字的使用**
	+ 用来代替某一个对象
	+ 可以调用一般属性或一般方法，放置在任何类成员中；
	+ 可以调用构造方法，只能放置在另一个构造方法内，只能放在该构造方法程序的第一行，***this()***
	+ 方法之间可以来回调用，语法不会出错，但运行是内存溢出；
	+ 构造方法不允许来回调用；

## Scanner 类的使用 ##
1. 调用jar包 java.util
2. 创建对象 Scanner input = new Scanner(System.in);
3. 使用方法 
	+ input.nextLine()  **以回车符作为截至，将回车符连同之前的所有字符都读取出来，将回车符扔掉，剩下字符组成一个完整的字符串，返还给我们**
	+ input.next() 同nextInt，看到回车或者空格都认为是结束符号；
	+ input.nextInt() **以回车符作为截至，将回车符之前的所有字符都读取出来，回车符留在队列中**
4. 利用包装类可以进行String与基本类型之间的转换
	int value = Integer.parseInt("123");

# 类与类之间关系 #
## 继承 ##
+ 每一个类都有继承类，默认为Object类，所有类都是直接或者间接继承Object，Object没有父类；包含以下方法：
	1. Object.toString()：打印输出时将对象变成String字符串
	2. Object.hashCode()：将对象在内存中的地址经过计算得到一个int值
	3. Object.getClass()：获取对象对应类的类映射
	4. Object.equals()：用来比较两个对象的内容，默认是==，比较地址
	5. Object.getName();
|区别|方法重写 override|方法重载 overload|
|:---:|:---:|:---:|
|类|产生两个继承关系的类 子类重写父类的方法|一个类中的一组方法|
|权限|子类可以大于等于父类|没有要求|
|特征|final：父类方法是final 子类不能重写 static:父类方法是static 子类不存在 abstract：父类方法是abstract 子类必须重写(子类是具体必须重写 子类是抽象类 可以不重写)|没有要求|
|返回值|子类可以小于等于父类|没有要求|
|名字|子类与父类一致|一个类中的好多方法名必须一致|
|参数|子类与父类一致|每一个方法的参数必须不一致(个数 类型 顺序)|
|异常|如果父类方法抛出运行时异常,子类可以不予理会; 如果父类方法抛出编译时异常,子类抛出异常的个数小于等于父类 子类抛出异常的类型小于等于父类|没有要求|
|方法体|子类的方法内容与父类不一致|每一个重载方法 执行过程不一致|

# 错误与异常 #
### Error错误 ###
   通常是一些物理性的，JVM虚拟机本身出现的问题，程序指令是处理不了的。
### Exception异常 ###
   通常是一种人为规定的不正常现象，是给定的程序指令产生了一些不符合规范的事情。
### Throwable ###
   + Error错误: StackOverflowError, OutOfMemoryError;
   + Exception异常: RuntimeException, IOException;
### 异常的分支体系 ###
+ 运行时异常/非检查异常
  + Error和RuntimeException都算是运行时异常；javac编译时不会提示发现异常，在程序编写时不要求必须处理；如果愿意可以添加处理手段(try  throws)；知道为何产生异常及如何处理；
  1. InputMisMatchException 输入不匹配
  2. NumberFormatException 数字格式化：int value = Integer.parseInt("abc");
  3. NegativeArraySizeException 数组长度负数
  4. NullPointerException 空指针异常
  5. ArithmeticException 数字异常：10/0 整数不允许除以0    Infinity小数除以0会产生无穷
  6. ClassCastException 造型异常 
  7. StringIndexOutOfBoundsException 字符串越界异常
  8. ArrayIndexOutOfBoundsException 数组越界异常
  9. IndexOutOfBoundsException 集合越界异常：List家族
  10. ILLegalArgumentException 非法参数异常
+ 编译时异常/检查异常
  + 除了Error和RuntimeException之外的异常，javac编译时强制要求处理异常(try or throws)，因为这样的异常在程序运行处理过程中极有可能产生问题，后续的所有执行就停止了
  1. InterruptException
### 添加处理异常的手段 ###
+ 处理异常指的是处理异常之后，*后续代码不会因为此异常而终止执行*
1.** try{} catch(){} [ finally{} ] **
  + try不能单独存在，后面必须添加catch或者finally；
  + catch之后有一组()，目的是捕获某一种异常
  + catch可以存在很多，捕获不同的异常；捕获的异常之间要么没有继承关系，要么从小到大进行捕获；
  + finally不是必须存在的，若存在finally，则必须执行；
  + 如果在方法内部有返回值，不管return关键字在哪里，finally一定会执行完毕，返回值的具体结果看情况；
2. throws抛出
   + 异常只能在方法内抛出，属性是不能处理异常的；
   + 多个异常逗号隔开；
   + 抛出的多个异常要么无继承关系，要么先抛出小异常，再抛出大异常；
### 自定义异常 ###
1. 自己描述一个异常类；
2. 类继承
   + 如果继承RuntimeException，运行时异常，不用添加处理手段；
   + 如果继承是Exception，编译时异常，必须添加处理手段；
3. 创建一个当前自定义类的对象，**通过throw关键字，主动产生异常**， throw new MyException();