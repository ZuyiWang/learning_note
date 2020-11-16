

# C++ Primer Plus Edition 6
## Chapter 1 预备知识
+ Bjarne Stroustrup 在20世纪80年代，在贝尔实验室发明了C++语言

## Chapter 2 C++初识
1. ***C++大小写敏感，区分大小写***
2.  ***所有语句都以分号结束***
```c++
#include <iostream>
using namespace std;

int main(){
	cout << "Come up and C++ me some time. " << endl;
	cout << "You won't regret it! " << endl;
	cin.get();
	return 0;
}
```
3. C++要求main()函数的定义以函数头`int main()`开始，该函数被启动代码调用，而启动代码由编译器添加到程序中；
4. 注释
   + 单行注释  双斜杠  //   **这是C++风格的注释**
   + 多行注释 ` /*  comment  */`  **这是C风格的注释**
5. 头文件名
   + C语言风格的头文件用扩展名`.h`结尾，而C++风格的头文件没有扩展名；
   + 有些C的头文件被转换为C++的头文件，去掉扩展名`.h`，并在文件名之前加上前缀`c`, 例如math.h====>cmath
6. 名称空间
   + 可以用来区别不同厂商封装的同名函数
   ```c++
   Microflop::wanda("go dancing");
   Piscine::wanda("a fish named Desire");
   ```
   + using编译指令可以编译整个名称空间，也可以编译具体的所需名称；
   ```c++
   using namespace std;
   // using namespace std::cout;
   // using namespace std::cin;
   ```
7. 变量与声明语句
   ```c++
   int carrots;
   ```
   + C++必须在首次使用变量前声明；
   + 一般C中所有的变量声明都位于函数的开始位置，但是C++中并无此要求
8. 赋值语句
   + C++中可以连续使用赋值运算符，和python类似
   ```c++
   int steinway;
   int baldwin;
   int yamaha;
   yamaha = baldwin = steinway = 88;
   ```
9. cin与cout的优势
   ```c++
   #include <iostream>
   using namespace std;
   
   int main(){
      int carrots;
   
      cout << "How many carrots do you have?" << endl;
      cin >> carrots;
      carrots = carrots + 2;
      cout << "Now you have " << carrots << "carrots." << endl;
      return 0;
   }
   ```
   + ***可以使用cout进行输出的拼接***， 利用运算符重载和函数重载；
10. 类简介
   + C++提供了两种调用对象特定函数的方法：
      1. 使用类方法，本质上是函数调用；
      2. 重新定义运算符，例如cout是重新定义运算符`<<`；
11. 函数简介
   + ***使用函数之前需要提供函数原型***，即参数类型与返回值类型
   ```c++
   double sqrt(double);  // function prototype
   // 结尾的分号说明函数原型是一个语句，而不是函数头
   double pow(double, double);
   int rand(void);
   ```
12. 多函数程序中使用using编译指令，存在以下四种情况：
   + `using namespace std;`放在函数定义之前，则文件中的所有函数都可以使用std中的元素；
   + `using namespace std;`放在特定的函数之中，则只有该函数中可以使用std中的元素；
   + `using namespace std::cout;`放在特定函数之中，则该函数中可以使用特定的元素；
   + 完全不使用using编译指令；

## Chapter 3 处理数据
1. C++的命名规则
   + 只能使用字母字符，数字和下划线；
   + 不可以数字开头；
   + 区分大小写；
   + ***以两个下划线或者下划线+大写字母打头的名称被保留给实现(编译器及其使用的资源)使用。以一个下划线开头的名称被保留给实现，用作全局标识符***
2. 整型
   + 4种符号类型的不同的整型宽度：
   
     | 类型      | 长度                         |
     | --------- | ---------------------------- |
     | short     | 至少16位                     |
     | int       | 至少与short一样长            |
     | long      | 至少32位，且至少与int一样长  |
     | long long | 至少64位，且至少与long一样长 |
   + 获取整型的长度 
      1. c++中1字节不一定等于8位，这与系统采用的字符集有关； 
      2. `sizeof`运算符可以返回类型或变量的长度，单位为字节；
      3. 头文件`climits`(limits.h)包含了整型限制的信息：`INT_MAX`为int的最大取值,`CHAR_BIT`为字节位数； 
      ```c++
      #include <iostream>
      #include <climits>
      
      int main(){
         using namespace std;
         int n_int = INT_MAX;
         short n_short = SHORT_MAX;
         long n_long = LONG_MAX;
         long long n_llong = LLONG_MAX;
         
         // sizeof运算符对计算需要加括号，对变量时括号可选
         cout << "int is " << sizeof (int) << "bytes" << endl;
         cout << "short is " << sizeof n_short << "bytes" << endl;
         
         cout << "Minimum int value = " << INT_MIN << endl;
         cout << "Bits per byte = " << CHAR_INT << endl;
         return 0;
      }
      ```
3. 变量初始化
   ```c++
   int uncles = 5;
   int aunts(6);   
   int hamburgers = {24};  //=可以省略
   int rheas{7};
   ```
4. 无符号类型
   + 使用关键字`unsigned`修饰声明
   ```c++
   unsigned short change;
   ```
5. 整型的进制转换
   + 八进制以数字'0'开头；
   + 十六进制以'0x'或'0X'开头；
   + cout的控制符`hex`、`oct`输出相应的进制数，***在修改进制之前，原来的进制一直有效***；
   ```c++
   int chest;
   cout << "chest = " << chest << " ( decimal )" << endl;
   cout << hex; // 16进制
   cout << "chest = " << chest << " ( hexadecimal )" << endl;
   cout << oct; // 8进制
   cout << "chest = " << chest << " ( octal )" << endl;
   ```
6. char类型——专门为存储字符而设计
   + 另一种整型，存储的是字符的数值编码，可以省略字符与ASCII编码之间的转换函数\
   + ***c++中字符用单引号括起，对字符串使用双引号***
   ```c++
   #include <iostream>
   
   int main(){
      using namespace std;
      char ch = ‘M’;
      int i = ch;   // i=77
      cout << "The ASSCII code for " << ch << " is " << i << endl;
      
      cout << "Add one to the character code: " << endl;
      ch = ch + 1;
      i = ch;
      cout << "The ASSCII code for " << ch << " is " << i << endl;
      
      cout << "Displaying char ch using cout.put(ch): ";
      cout.put(ch);
      cout.put('!');
      cout << endl << "Done" << endl;
      return 0;
   }
   ```
   + 转义序列，为了表示一些不能直接通过键盘输入到程序的字符，例如换行符等；
   + 转义字符与处理常规字符相同：作为字符常量时单引号括起，放在字符串中时不要使用单引号；
   + 可以使用八进制与十六进制的数字转义序列来表示字符；例如'M'的ASCII码是77，其八进制与十六进制是\095和\x4d，则'\095'和'\x4d'可以表示字符常量'M'；但是这种方式与具体的编码方式与编码数值有关，因此不推荐；
   + 可以显示的定义char类型是否有符号`signed char`或者`unsigned char`，正常情况下char既不是有符号也不是没符号，这与硬件属性有关。
   + `wchar_t`宽字符类型可以表示扩展字符集，这种类型的输入输出需要使用`wcin`和`wcout`，通过在前缀加上'L'可以表示宽字符常量和宽字符串。
   ```c++
   wchar_t bob = L'P';
   wcout << L"tall" << endl;
   ```
   + char16_t和char32_t， 前者长16位，后者长32位，都是无符号的；其字符常量的前缀分别为'u'和'U'
7. const限定符
   + 需在声明时进行初始化
   + 优点：可以声明具体类型和一些复杂类型，可以确定常量的作用域；
8. 浮点数  
   + 书写方式：`12.34`或者`2.52e+8, 8.33E-4`，可以使用e或者E，'+'可以省略；
   | 类型      | 长度                         |
   | --------- | ---------------------------- |
   | float     | 至少32位                     |
   | double    | 至少48位，且不少于float       |
   |long double| 至少与double一样长            |
   + 在cfloat或float.h文件中可以找到浮点数的相关限制；
   
   + 浮点常量(8.24, 2.4E8)默认存储为***double***类型，***如果希望存储为float类型，请使用'f'或'F'后缀，对于long double类型可以使用'l'或'L'结尾***
   
   + 浮点数的优缺点
   
     | 优点                                 | 缺点                         |
     | ------------------------------------ | ---------------------------- |
     | 可以表示小数，非常大的数和非常小的数 | 运算速度比整数慢，且精度降低 |
9. 类型转换
   + 初始化和赋值的时候会进行类型转换，有时候会发生精度降低的情况，有时候会是随机数值；c++中用大括号的初始化称为列表初始化，它对类型转换的要求很严格，必须保证初始化是不会有精度降低的。
   + 表达式中的类型转换
      1. 整型提升， 计算表达式时候 bool, char, unsigned char, signed char, short 数值转换为int；wchar_t将被提升为下列类型中第一个足够存储的：int，unsigned int，long， unsigned long；
      2. 运算涉及两种类型的计算时，较小的类型转换为较大的类型
   + 函数传递参数时的转换，通常由函数原型控制；
   + 强制类型转换
   ```c++
   int thorn;
   
   long (thorn);  //return a type long conversion of thorn, C++ style
   (long) thorn;  //return a type long conversion of thorn, C style
   static_cast<long> (thorn);  //return a type long conversion of thorn
   ```
10. auto关键字
   + 根据变量的初始赋值推断变量的类型
   ```c++
   auto n = 100;  //n is int
   auto x = 1.5;  //x is double
   auto y = 1.3e12L;  //y is long double
   ```
## Chapter 4 复合类型
1. 数组
   1. 数组初始化
      + 只有定义数组时可以初始化，此后则不能初始化，不能将一个数组赋给另一个数组；
      + 只对数组的一部分初始化，则编译器将把其他元素设置为0；
      + 如果初始化[]为空，则编译器将计算元素个数；
      ```c++
      int cards[4] = {3, 6, 8, 10};
      
      float hotel[5] = {1.5, 2.0};
      long totals[10] = {0};  //设置所有元素为0
      
      short things[] = {1, 5, 3, 8};
      ```
      + c++中初始化数组可以省略'='；
      + 可以不再{}中包含任何东西，这将把所有元素都设置为零；
2. 字符串
   + 字符串以空字符结尾，空字符写作\0，其ASCII码为0，用来标记字符串的结尾
   + 字符数组初始化为字符串：双引号括起字符串，这种字符串叫做字符串常量或字符串字面值
   1. 字符串常量的拼接：任何两个由空白(空格、制表符和换行符)分隔的字符串常量都将自动拼接成一个；拼接时第二个字符串紧接着第一个字符串，中间没有空格
   ```c++
   cout << "I'd give my right arm to be" " a great violinist.\n"
   ```
   2. `sizeof`运算符计算的是整个数组的长度，`strlen()`返回的是存储在数组中的字符串的长度，***strlen()只计算可见的字符，不把空字符计算在内***
   3. 字符串输入：每次读取一行字符串输入`cin.get()`和`cin.getline()`;
      + `cin.getline()`会***读取并丢弃输入的换行符***，接受两个参数，第一个参数为存储输入行的数组名，第二个参数为读取的字符数，如果这个参数是20，则函数最多读19个字符，然后自动添加一个空字符；getline()函数在读取指定数目的字符或遇到换行符时停止读取；
      ```c++
      const int ArSize = 20;
      char name[ArSize];
      char dessert[ArSize];
      
      cin.getline(name, ArSize);
      cin.getline(dessert, ArSize);
      // 也可以连起来调用getline函数
      // cin.getline(name, ArSize).getline(dessert, ArSize);
      ```
      + `cin.get()`的参数设置与`cin.getline()`相同，但是它将输入的换行符保存在输入队列中，所以需要多一次调用接受换行符；
      ```c++
      cin.get(name, ArSize);
      cin.get();
      cin.get(dessert, Arsize);
      // 也可以连起来调用get函数
      // cin.get(name, ArSize).get();
      ```
   4. 空行问题
   + 当get()读取空行后，将会设置失效位，接下来的输入将被阻断，可以使用以下命令`cin.clear()`来回复
   + 当输入行的字符比指定的多，则getline()和get()将把余下的字符留在输入队列中，而getline()还会设置失效位，关闭后面输入；
   ```c++
   // 混合输入方式
   #include <iostream>
   
   int main(){
      using namespace std;
      
      int year;
      char address[80];
      
      cout << "what year was your house built?" << endl;
      cin >> year;
      cin.get();  //必不可少，接受换行符
      // or cin.get(ch)  (cin >> year).get()   (cin >> year).get(ch)  
      cout << "what is its street address?" << endl;
      cout << "Year built: " << year << endl;
      cout << "Address: " << address << endl;
      cout << "Done" << endl;
      return 0;
      
   }
   ```
3. 

​     


​     
