# Python 基础教程学习笔记
---
## Chapter 1 基础知识
   + python程序的末尾不用添加分号，添加也不会有影响
   + 数的计算和表达式
      + '/'代表除法，'//'代表整除，丢弃小数部分,例如`5.0//2.4=2.0`
      + ***除法结果默认是浮点数***
      + '//'是**向下圆整**，例如`-10//3=-4, -10//-3=3`
      + 求余运算, `x % y=x - (x//y)*y`, 该运算符可以用于浮点数或者负数
         ```
         10 % 3 = 1
         10 % -3 = -2 = 10 - (10//-3)*(-3) = 10 - (-4)*(-3)
         -10 % 3 = 2
         -10 % -3 = -1
         ```
      + 乘方运算 **, 优先级高于单目运算符
   + 进制
      + 十六进制 `0xAF`
      + 八进制 `0o10`
      + 二进制 `0b1011`
   + 变量
      + python语言没有默认值, 使用变量前必须进行赋值
      + 变量名称由**数字+字母+下划线组成, 数字不能打头**
   + 语句
      + 输出语句 print在3版本中是函数, 因此必须`print()`, 但是在2版本中, print是一个语句, 不需要圆括号
      + 输入语句 input函数, 圆括号中可以放一些提示符, 用双引号包围
        `x=input("x: ")`
   + 函数
      + `pow(2, 3)=8`
      + round()是就近圆整, 在两个整数一样近时圆整到偶数
      + floor()是向下圆整
      + ceil()是向上圆整
      + 可以使用变量来引用函数
      ```
      import math
      foo = math.sqrt
      foo(4)
      ```
   + 模块
      + 通过import来导入模块
      ```
      import math
      math.floor(32.9)
      ```
      + 确定不会从不同模块中导入多个同名函数时, 可以如下导入, 省略使用时的模块前缀
      ```
      from math import sqrt
      sqrt(9)
      ```
      + cmath模块用来处理复数, 虚数以j结尾
      ```
      import cmath
      cmath.sqrt(-1)
      ```
      + __future__模块, 可以导入当前版本不支持, 但是未来版本称为标准组成部分的功能
   + 脚本运行
     
      + `#!/usr/bin/env python`寻找相应的解释器运行程序
   + 添加注释
     
      + `# 后面的文字将被忽略`
   + 字符串
      + 可以用单引号或者双引号引用字符, 没有差别
      + **转义字符**, 使用反斜杠(`\`)
      + 字符串可以像普通数字一样进行相加
      + print()函数在打印时不会用引号包围结果, 但是直接输出时会有引号
      ```
      >>> "Hello, world!"
      'Hello, world!'
      >>> print("Hello, world!")
      Hello, world!
      ```
   + str函数和repr函数
      + str()输出会进行转义，repr()输出会保留圆括号中原有合法字符
      ```
      >>> print(repr("Hello, \nworld!"))
      'Hello, \nworld!'
      >>> print(str("Hello, \nworld!"))
      Hello,
      world!
      ```
   + 长字符串
      + 三引号可以表示长字符串, 跨越多行
      + 普通字符串的行尾加反斜杠也可跨越多行, 反斜杠+换行符将被转义, 即忽略
      ```
      print("Hello, \
      world!")
      ```
   + 原始字符串
      + 不对反斜杠做特殊处理, 让字符串的每个字符保持原样
      + 用前缀r表示
      ```
      >>> print(r"C:\nowhere")
      C:\nowhere
      >>> print(r"Let\'s go!")
      Let\'s go!
      ```
      + 原始字符串不能以单个反斜杠结尾, 除非对其进行转义, 即双斜杠结尾
      ```
      >>> print(r'This is illegal\\')
      This is illegal\\
      ```
      + 如果希望单斜杠结尾, 则需要将单斜杠进行转义, 然后 作为单独字符串进行拼接
      ```
      >>> print(r'This is illegal'+'\\')
      This is illegal\
      ```
   + 编码
     
      + 字符使用unicode码点进行编码, 默认是UTF-8

## Chapter 2 列表和元组
   + 列表和元组的主要区别: **列表可以修改, 元组不可修改**
   + 列表元素放在方括号内, 并用逗号隔开, ***列表中元素不一定同类型***
   ```
   >>> edward = ['Edward Gumby', 42]
   ```
   + 序列操作
      + 索引
         1. 元素编号从0开始, 负数索引代表从右往左数, -1是最后一个元素位置
         ```
         >>> greeting = "Hello"
         >>> greeting[0]
         'H'
         >>> greeting[-1]
         'o'
         ```
         2. 可以对字符串直接进行索引, 无需进行赋值
         ```
         >>> 'Hello'[1]
         'e'
         ```
      + 切片: 访问一个特定范围内的元素
         1. 可以使用两个索引, 并用冒号分隔, 第一个索引指定的元素包含在切片中, 但第二个索引指定的元素不在切片内, ***切片的时候不会报错IndexError***
         ```
         >>> numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
         >>> numbers[3:6]
         [4, 5, 6]
         >>> numbers[0:1]
         [1]
         >>> numbers[-3:-1]
         [8, 9]
         ```
         2. 切片如果从头开始, 可以省略第一个索引, 如果到序列的末尾结束, 可以省略第二个索引
         ```
         >>> numbers[:3]
         [1, 2, 3]
         >>> numbers[-3:]
         [8, 9, 10]
         >>> numbers[:]
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
         ```
         3. 切片可以指定步长, **步长为负数代表从右向左提取元素, 步长为正数时从起点移到终点, 步长为负数时从终点移到起点**
         ```
         >>> numbers[0:10:1]
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
         >>> numbers[0:10:2]
         [1, 3, 5, 7, 9]
         >>> numbers[8:3:-1]
         [9, 8, 7, 6, 5]
         >>> numbers[10:0:-2]
         [10, 8, 6, 4, 2]
         ```
      + 相加
         + 加法运算可以拼接序列
         + 数字序列和字符串序列不可拼接
      + 相乘
         1. 序列与数x相乘, 将重复序列x次来创建一个新序列
         2. 可以用None来初始化一个空列表
         ```
         >>> 'python' * 5
         'pythonpythonpythonpythonpython'
         >>> sequence = [None] * 5
         >>> sequence
         [None, None, None, None, None]
         ```
      + 元素检查
        
         1. 检查特定的值是否包含在序列中, `in`, 返回结果为布尔值
      + 序列长度、最大值、最小值
        
         1. len(), max(), min()
   + 列表
      + 序列的一种, 可以修改其中元素
      + 函数list
         1. 可以将任何序列转换为列表
         ```
         >>> list('Hello')
         ['H', 'e', 'l', 'l', 'o']
         ```
      + 赋值与删除元素
         1. 普通赋值语句
         2. del语句可以删除元素
         ```
         >>> x = [1, 1, 1]
         >>> x[1] = 2
         >>> x
         [1, 2, 1]
         >>> del x[0]
         >>> x 
         [2, 1]
         ```
      + 切片赋值与删除
         1. 替换的切片长度与原长度可以不一致, **因此此操作可以修改列表长度**
         2. 可以用空切片和原切片进行替换, **可以实现插入新元素或者删除元素**
         3. ***新切片的替换必须是列表形式, 不能是单个元素***
         ```
         >>> name = list('Perl')
         >>> name[1:] = list('ython')
         >>> name 
         ['P', 'y', 't', 'h', 'o', 'n']
         >>> numbers = [1, 5]
         >>> numbers[1:1] = [2, 3, 4]
         >>> numbers 
         [1, 2, 3, 4, 5]
         >>> numbers[1:4] = []
         >>> numbers 
         [1, 5]
         ```
      + append方法
         1. 将对象添加到列表末尾
         2. append()圆括号中是添加对象, 会将其作为一个整体添加到原列表的末尾
         ```
         >>> numbers = [1, 2, 3]
         >>> numbers.append(4)
         >>> numbers
         [1, 2, 3, 4]
         >>> numbers.append([5, 6])
         >>> numbers
         [1, 2, 3, 4, [5, 6]]
         ```
      + clear方法
        
         1. 就地清空列表内容
      + copy方法
         1. 常规的赋值语句是地址的复制, 没有新列表产生, 两个变量指向同一个列表
         2. copy方法, 产生一个原列表的新副本
         3. `a[:]`或者`list(a)`也可以产生新副本
         ```
         >>> a = [1, 2, 3]
         >>> b = a
         >>> b[1] = 4
         >>> a
         [1, 4, 3]
         >>> a = [1, 2, 3]
         >>> b = a.copy
         >>> b[1] = 4
         >>> a
         [1, 2, 3]
         ```
      + count方法
        
         1. 计算指定元素在列表中出现的次数
      + extend方法
         1. 同时将多个元素添加到列表的末尾
         2. **extend()圆括号中是添加的序列, 不能是单个元素**
         3. extend方法是修改原列表, 而相加得到的列表拼接是返回一个新列表
         ```
         >>> a = [1, 2, 3]
         >>> b = [4, 5, 6]
         >>> a + b
         [1, 2, 3, 4, 5, 6]
         >>> a
         [1, 2, 3]
         >>> a.extend(b)
         >>> a
         [1, 2, 3, 4, 5, 6]
         ```
      + index方法
        
         1. 查找列表中指定值第一次出现时的索引, 如果指定值不存在则会报错
      + insert方法
         1. 将一个对象插入列表中的任意位置
         2. insert()圆括号中是添加对象, 会将其作为一个整体添加到原列表, 类似于append()
         ```
         >>> numbers = [1, 2, 3, 4, 5]
         >>> numbers.insert(3, 56)
         >>> numbers
         [1, 2, 3, 56, 4, 5]
         >>> numbers.insert(2, [45, 46])
         >>> numbers
         [1, 2, [45, 46], 3, 56, 4, 5]
         ```
      + pop方法
         1. 从列表中删除一个元素, 并将该元素返回, 默认是最后一个元素
         2. 圆括号中是要删除元素的索引
         ```
         >>> x = [1, 2, 3]
         >>> x.pop()
         3
         >>> x
         [1, 2]
         >>> x.pop(0)
         1 
         ```
      + remove方法
        
         1. 用于删除第一个为指定值的元素， 如果参数在列表中不存在, 则报错
      + 列表排序
         + reverse方法
            1. 以相反顺序排列原来列表, 修改原列表, 不返回值
            2. **是原来列表的相反顺序, 不是元素逆序**
            ```
            >>> x = [1, 2, 3]
            >>> x.reverse()
            >>> x
            [3, 2, 1]
            ```
         + sort方法
            1. 对原列表进行修改, 使元素按顺序排列, 无返回值
            ```
            >>> x = [4, 6, 2, 1, 7, 9]
            >>> x.sort()
            >>> x
            [1, 2, 4, 6, 7, 9]
            ```
         + sorted方法
            1. 对原列表中的元素进行顺序排列, 并返回该新列表, 原列表不发生改变
            ```
            >>> x = [4, 6, 2, 1, 7, 9]
            >>> y = x.sorted()
            >>> x
            [4, 6, 2, 1, 7, 9]
            >>> y
            [1, 2, 4, 6, 7, 9]
            ```
            2. sorted方法可以用于任何序列, 不只是列表, 但总是返回一个列表
            ```
            >>> sorted('Python')
            ['P', 'h', 'n', 'o', 't', 'y']
            ```
         + 逆序排列
           
            1. 可以对列表先调用sort或者sorted方法, 再调用reverse方法
         + 高级排序
            1. sort函数与sorted函数可以接受参数key, reverse
            2. key为设置的排序函数, 使用该函数为为每一个元素创建一个键, 然后根据该键对元素进行排序
            3. 元素是逆序排列还是顺序排列, reverse设置为True或者False
            ```
            >>> x = ['aadvark', 'abalone', 'acme', 'add', 'aerate']
            >>> x.sort(key=len)
            >>> x
            ['add', 'acme', 'aerate', 'abalone', 'aadvark']
            ```
   + 元组
      + 不可修改的序列
      + 元组的创建用圆括号括起， 用逗号分隔元素, ***也可以省略圆括号, 只用逗号分隔元素***
      + **单个元素的元组, 也必须用逗号分隔, 为了和单个数字区分**
      + tuple函数, 参数为一个序列, 返回结果为元组
      ```
      >>> 3 * (40+2)
      126
      >>> 3* (40+2,)
      (42, 42, 42)
      >>> tuple('abc')
      ('a', 'b', 'c')
      ```

## Chapter 3 字符串
   + 序列的一种, 适用所有的标准序列操作, 但是序列是不可变的, 因此不可赋值切片
   + 设置字符串格式
      + 使用百分号%, 在%左边指定一个字符串, 并在右边指定要设置其格式的值
         1. 指定其设置的格式的值, 可以使用单个值, 或者元组、字典
         2. `%s`代表将值设置为字符串, `%.3f`代表将值设置为保留3位小数的浮点数
         ```
         >>> format = "Hello, %s. %s enough for ya?"
         >>> values = ('world', 'Hot')
         >>> format % values
         "Hello, world. Hot enough for ya?"
         ```
      + 使用字符串方法format()
         1. 字符串中需要替换的部分用大括号括起, 大括号中包含可能的关键字名称、索引或者格式设置信息
         2. 大括号内设置格式, `关键字:格式设置`
         ```
         >>> "{}, {} and {}".format("first", "second", "third")
         'first, second and third'
         >>> "{3} {0} {2} {1} {3} {0}".format("be", "not", "or", "to")
         'to be or not to be'
         >>> from math import pi
         >>> "{name} is approximately {value:.2f}.".format(value=pi, name="pai")
         'pai is approximately 3.14.'
         ```
         3. 关键字与替换变量同名, 可以在字符串前面加上f
         ```
         >>> from math import e
         >>> "Euler's constant is roughly {e}.".format(e=e)
         "Euler's constant is roughly 2.718281828459045."
         >>> f"Euler's constant is roughly {e}."
         "Euler's constant is roughly 2.718281828459045."
         ```
   + 字符串方法
      + center方法
         1. 两边添加字符让字符串居中, 共两个参数, 包括填充后字符串长度和填充字符
         ```
         >>> "The Middle by Jimmy Eat World".center(39) 
         '     The Middle by Jimmy Eat World     '
         >>> "The Middle by Jimmy Eat World".center(39, '*')
         '*****The Middle by Jimmy Eat World*****'
         ```
      + find方法
         1. 在字符串中查找子串, 返回子串第一个字符的索引, 否则返回-1
         2. 可以指定字符串中搜索的起点与终点的索引, 搜索范围包含起点索引字符不包含终点索引字符
         ```
         >>> subject = "$$$ Get rich now!!! $$$"
         >>> subject.find('!!!')
         16
         >>> subject.find('!!!', 1, 16)
         -1
         ```
      + join方法
         1. 用于合并序列的元素
         2. `分隔符.join(合并序列)`
         3. 合并序列的元素必须都是字符串
         ```
         >>> seq = ['1', '2', '3', '4', '5']
         >>> '+'.join(seq)
         '1+2+3+4+5'
         >>> '+'.join('python')
         'p+y+t+h+o+n'
         >>> '+'.join([1, 2, 3, 4, 5])
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
         TypeError: sequence item 0: expected str instance, int found
         ```
      + split方法
         1. 将字符串拆分为序列
         2. 参数为为指定分隔符, 默认为单个或多个连续空白字符（空格, 制表符, 换行符）
         ```
         >>> '1+2+3+4+5'.split('+')
         ['1', '2', '3', '4', '5']
         ```
      + lower方法
        
         1. 返回字符串的小写版本
      + replace方法
         1. 将指定子串都替换为另一个字符串, 并返回替换后的结果
         2. `replace(被替换字符串, 替换字符串)`
         ```
         >>> "This is a test".replace('is', 'eez')
         'Theez eez a test'
         ```
      + strip方法
         1. 在字符串的开头和结尾删除指定字符, 默认是空白字符
         ```
         >>> "*** SPAM * FOR * EVERYONE!!! ***".strip(" *!")
         'SPAM * FOR * EVERYONE'
         ```
      + translate方法
         1. 可以根据table, 将字符串的特定子串进行转换
         2. 该方法只能接受单字符替换, 因此可以同时替换多个字符
         3. table的创建调用str.maketrans(), 接受三个参数, 被替换字符, 替换字符, 删除字符
         ```
         >>> table = str.maketrans('cs', 'kz')
         >>> table
         {99: 107, 115: 122}
         >>> 'this is an incredible test'.translate(table)
         'thiz iz an inkredible tezt'
         >>> table = str.maketrans('cs', 'kz', ' ')
         >>> 'this is an incredible test'.translate(table)
         'thizizaninkredibletezt'
         ```

## Chapter 4 字典
   + 通过名称来访问其各个值, 映射结构
   + 创建字典
      + 字典的key必须是唯一的, 对字典使用`in`是检查字典中是否存在相应key
      + 列举 
         1. 用大括号括起来, `key: value`, 不同项之间逗号隔开
         ```
         >>> phonebook = {'Alice': '2341', 'Beth': '9102', 'Cecil': '3258'}
         ```
      + 函数dict
         1. 通过其他映射或者键值对序列创建字典
         2. 传入关键字实参创建字典
         ```
         >>> items = [('name', 'Gumby'), ('age', '42')]
         >>> d = dict(items)
         >>> d
         {'name': 'Gumby', 'age': '42'}
         >>> a = dict(name='Gumby', age=42)
         >>> a
         {'name': 'Gumby', 'age': 42}
         ```
      + 字典方法copy()
   + 字典的基本操作
      + 类似序列, len(), del, k in d以及键值对的查找与赋值
      + key的类型可以是整数、字符串、元组, 但是确定之后不可变
      + 字典中不存在的key, 可以直接给它赋值, 从而创建新的一项
   + 字典用于字符串格式设置
      + 使用函数format_map(), 传入参数字典
      ```
      >>> phonebook
      {'Alice': '2341', 'Beth': '9102', 'Cecil': '3258'}
      >>> "Cecil's phone number is {Cecil}.".format_map(phonebook)
      "Cecil's phone number is 3258."
      ```
   + 字典方法
      + clear方法
        
         1. 删除所有的字典项, 修改原来的字典, 无返回值
      + copy与deepcopy方法
         1. copy()返回一个新字典, 包含原有字典的键值对, 这种方法是浅复制, 因为值本身是原件而非副本
         ```
         >>> x = {'username': 'admin', 'machine': ['foo', 'bar', 'baz']}
         >>> y = x.copy()
         >>> y['username'] = 'mlh'
         >>> y['machine'].remove('bar')
         >>> x
         {'username': 'admin', 'machine': ['foo', 'baz']}
         >>> y
         {'username': 'mlh', 'machine': ['foo', 'baz']}
         ```
         2. 模块copy中的deepcopy执行的深复制, 同时复制key及其value
         ```
         >>> from copy import deepcopy
         >>> d = {}
         >>> d['names'] = ['Alfred', 'Bertrand']
         >>> c = d.copy()
         >>> dc = deepcopy(d)
         >>> d['names'].append('Clive')
         >>> c
         {'names': ['Alfred', 'Bertrand', 'Clive']}
         >>> dc
         {'names': ['Alfred', 'Bertrand']}
         ```
         3. 针对Python中简单对象的复制，copy和deepcopy没有什么区别，就是和大家通常理解的复制是一样的，在内存中新开辟一个空间，将原来地址中的数据拷贝到新的地址空间中
         ```
         >>> import copy
         >>> a = [1, 2, 3, 4]
         >>> b = copy.copy(a)
         >>> c = copy.deepcopy(a)  
         >>> print(a == b)
         True
         >>> print(a is b)
         False
         >>> print(a == c)
         True
         >>> print(a is c)
         False
         ```
         4. 复杂对象可以理解为另外包含其他简单对象的对象，也就是包含子对象的对象,例如List中嵌套List，或者Dict中嵌套List等, 对于复杂对象中的简单数据部分，无论是深复制还是浅复制，我们可以看到，Python都是采用的直接在内存中开辟新的地址空间，然后将值复制到新的地址空间。对于复杂对象的子对象部分来说：深复制是在内存中开辟一个新的空间，并且将子对象复制到新的地址空间，但是对于浅复制而言，我们可以看到并没有对子对象来开辟空间，新复制的对象和原来的对象同时指向了同一个List对象（也就是同一个对象的引用）
         ```
         >>> import copy
         >>> a = {'name': 'test', 'age': 56, 'address': [1, 2, 3, 4, 5]}
         >>> b = copy.copy(a)
         >>> print(a is b)
         False
         >>> print(a['address'] is b['address'])
         True
         >>> c = copy.deepcopy(a)
         >>> print(a is c)
         False
         >>> print(a['address'] is c['address'])
         False
         ```
      + python的数据存储方式
         1. Python存储变量的方法跟其他OOP语言不同, 它与其说是把值赋给变量, 不如说是给变量建立了一个到具体值的reference
         2. 当在Python中a = something应该理解为给something贴上了一个标签a, 当再赋值给a的时候，就好象把a这个标签从原来的something上拿下来, 贴到其他对象上, 建立新的reference
         ```
         >>> a = [1, 2, 3]
         >>> b = a
         >>> a = [4, 5, 6] //赋新的值给 a
         >>> a
         [4, 5, 6]
         >>> b
         [1, 2, 3]
         ```
      a的值改变后，b并没有随着a变

         >>> a = [1, 2, 3]
         >>> b = a
         >>> a[0], a[1], a[2] = 4, 5, 6 //改变原来 list 中的元素
         >>> a
         [4, 5, 6]
         >>> b
         [4, 5, 6]
         # a的值改变后，b随着a变了
         ```
         
         ```
      + fromkeys方法
         1. 创建一个新字典, 包含指定的key, 对应的value = None
         2. 可以使用`{}.fromkeys()`或者`dict.fromkeys()`
         3. value也可以提供一个默认值, 则每一个key都是该值
         ```
         >>> dict.fromkeys(['name', 'age'], 'unknown')
         {'name': 'unknown', 'age': 'unknown'}
         ```
      + get方法
         1. 查找字典中的键值对, 与普通索引不同的是key不存在时不会报错, 可以返回None或设定值
         ```
         >>> d = {}
         >>> d['name']
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
         KeyError: 'name'
         >>> print(d.get('name'))
         None
         >>> d.get('name', 'NA')
         'NA'
         ```
      + items方法
         1. 返回一个包含所有字典项的***字典视图***, 其中每个元素为(key, value), 项的排列顺序不确定
         2. 字典视图可以用于迭代, 也可以进行长度和成员资格(in)检查, 字典视图是字典情况的实时反映, 随着字典value改变, 字典视图也自动改变
         3. 可以通过list()将字典视图复制到列表
         ```
         >>> a = {'name': 'test', 'age': 56, 'address': 5}
         >>> a.items()
         dict_items([('name', 'test'), ('age', 56), ('address', 5)])
         >>> it = a.items()
         >>> len(it)
         3
         >>> ('address', 5) in it
         True
         >>> a['address'] = 6
         >>> it
         dict_items([('name', 'test'), ('age', 56), ('address', 6)])
         >>> list(it)
         [('name', 'test'), ('age', 56), ('address', 6)]
         ```
      + keys方法
         1. 返回一个字典视图, 包含字典所有的key
         ```
         >>> a.keys()
         dict_keys(['name', 'age', 'address'])
         ```
      + values方法
         1. 返回一个字典的value组成的字典视图, 其可能包含重复的值
         ```
         >>> a = {'name': 'test', 'age': 56, 'address': 186, 'phone': 186}
         >>> a.values()
         dict_values(['test', 56, 186, 186])
         ```
      + pop与popitem方法
         1. pop()获取与指定键相关联的值, 并将该键值对从字典中删除
         ```
         >>> d = {'x': 1, 'y': 2}
         >>> d.pop('x')
         1
         >>> d
         {'y': 2}
         ```
         2. popitem()随机的获取一项, 并删除该项
         ```
         >>> a = {'name': 'test', 'age': 56, 'address': 5}
         >>> a.popitem()
         ('address', 5)
         >>> a
         {'name': 'test', 'age': 56}
         ```
      + setdefault()方法
         1. 根据key获取字典中相应的value, 与get()类似
         2. key不存在时, 可以以指定value添加到字典中, 并返回该值, 默认为None
         ```
         >>> d = {}
         >>> d.setdefault('name', 'N/A')
         'N/A'
         >>> d
         {'name': 'N/A'}
         >>> d['name'] = 'wzy'
         >>> d.setdefault('name', 'N/A')
         'wzy'
         >>> d 
         {'name': 'wzy'}
         ```
      + update方法
         1. 使用一个字典来更新另一个字典
         2. 被更新字典中不存在的项则添加, 存在key但value不同的项则更新value
         ```
         >>> a = {'name': 'test', 'age': 56, 'address': 5}
         >>> b = {'name': 'wzy', 'phone': 186}
         >>> a.update(b)
         >>> a 
         {'name': 'wzy', 'age': 56, 'address': 5, 'phone': 186}
         ```

## Chapter 5 条件循环及其他语句
   + print和import语句
      + print()打印多个参数
         1. 括号内可以同时传入多个参数, 条件是用逗号分隔
         2. 打印结果会在多个参数之间自动插入一个空格
         3. 可以设定参数sep和end修改自动分隔符与字符结尾符(默认是换行符)
         ```
         >>> print('age:', 25)
         age: 25
         >>> name = 'wzy'
         >>> salutation = 'Mr.'
         >>> greeting = 'Hello,'
         >>> print(greeting, salutation, name)
         Hello, Mr. wzy
         >>> print(greeting, salutation, name, sep='_')
         Hello,_Mr._wzy
         >>> print("Hello,", end='***')
         ··· print('world!')
         Hello,***world!
         ```
      + import导入重命名
         1. 一般导入方式
         ```
         >>> import math
         >>> from math import sqrt, open, pow
         >>> from math import * 
         ```
         2. 有多个重名模块或函数时, import语句末尾+as+指定别名
         ```
         >>> import module1 import open as open1
         >>> import module2 import open as open2
         ```
   + 赋值语句
      + 序列解包
         1. 序列1 = 序列2, 则将序列2中的元素的值依次赋值给序列1
         ```
         >>> x, y, z = 1, 2, 3
         >>> print(x, y, z)
         1 2 3
         >>> values = 4, 5, 6
         >>> x, y, z = values
         >>> print(x, y, z)
          4 5 6
         >>> b = {'name': 'wzy'}
         >>> key, value = b.popitem()
         >>> key
         'name'
         >>> value
         'wzy'
         ```
         2. 序列1和序列2的长度相同, 当长度不同时, 可以使用星号变量来收集多余的值
         3. 星号变量可以放在任意位置, 收集开头、中间、结尾的多余变量
         4. 序列2可以是任意类型, 但星号变量的收集结果总是列表
         ```
         >>> c = [1, 2, 3, 4]
         >>> x, y, *rest = c
         >>> rest
         [3, 4]
         >>> first, *rest, last = c
         >>> rest
         [2, 3]
         >>> a, *b, c = 'abc'
         >>> b
         ['b']
         ```
      + 链式赋值
         1. 多个变量关联到一个值
         ```
         >>> x = y = z = 100
         >>> x is y
         True
         ```
      + 自加自乘简写
         1. `+=`, `*=`, `/=`, `-=`等标准运算
         ```
         >>> st = 'foo'
         >>> st += 'bar'
         >>> st *= 2
         >>> st
         'foobarfoobar'
         ```
   + 代码块
      + 相同代码块有相同的缩进
      + 使用冒号表示接下来是代码块
   + 条件语句
      + 假值
        
         1. False, None, 0以及各中空序列、空映射
      + 条件语句
         1. if+elif+else
         ```
         if 条件1: 
            block1
         elif 条件2: 
            block2
         else: 
            block3
         ```
         2. 代码块嵌套, 可以条件嵌套条件
      + 比较运算符
         1. 支持链式比较, `0<age<50`
         2. is运算符, 检查两个对象是否相同, 而不是相等, 即是否是一个内存地址
         3. in运算符, 成员资格检查
         4. 字符串比较是根据字符的排列顺序, 字符都是unicode编码, 是根据码点排列的, 获取字符的顺序值用函数ord(), chr()根据顺序值返回字符
         ```
         >>> 'alpha' < 'beta'
         True
         >>> ord('B')
         66
         >>> ord('a')
         97
         >>> 'a' < 'B'
         False
         >>> chr(128586)
         '🙊'
         ```
         5. 序列的比较, 从第一个元素开始比较, 直到分出大小, 如果序列元素中有子序列, 则子序列也从第一个元素开始比较
         ```
         >>> [1, 2] < [2, 1]
         True
         >>> [2, [1, 4]] < [2, [1, 5]]
         True
         ```
      + 布尔运算符
         1. and, not, or
         2. 短路逻辑, 只做必要的计算
         ```
         x and y # 当x为假, 则返回x; 当x为真, 则返回y
         x or y # 当x为真, 则返回x; 当x为假, 则返回y
         ```
      + 断言
         1. assert 条件, "错误说明"
         2. 当条件不成立的时候, 抛出错误
         ```
         >>> age = 10
         >>> assert 0<age<100, "The age must be realistic"
         >>> age = -1
         >>> assert 0<age<100, "The age must be realistic"
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
         AssertionError: The age must be realistic
         ```
   + 循环语句
      + while循环, 条件满足时一直执行
      ```
      while 条件:
         block
      ```
      + for循环
         1. 针对序列或者可迭代对象, 遍历其中每一个元素
         ```
         >>> words = ['this', 'is', 'an', 'ex', 'parrot']
         >>> for word in words:
         ···    print(word)
         this
         is
         an
         ex
         parrot
         ```
         2. 创建范围函数range(起点, 终点), 范围只包含起点不包括终点
         3. range(终点)只有一个参数时看作是终点, 并且默认起点为0
         ```
         >>> range(0, 10)
         range(0, 10)
         >>> list(range(0, 10))
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
         ```
         3. 迭代字典
         4. 迭代字典时可以在for循环中使用序列解包
         ```
         >>> d = {'x':1, 'y':2, 'z':3}
         >>> for key in d:
         ···    print(key, "corresponds to", d[key])
         x corresponds to 1
         y corresponds to 2
         z corresponds to 3
         >>> for key, value in d.items():
         ···    print(key, "corresponds to", value)
         x corresponds to 1
         y corresponds to 2
         z corresponds to 3
         ```
      + 并行迭代
         1. zip(序列1, 序列2)将两个序列的元素依次组合, 返回一个元组组成的序列
         ```
         >>> names = ['anne', 'beth', 'george', 'damon']
         >>> age = [12, 45, 32, 102]
         >>> for name, age in zip(names, age):
         ···    print(name, 'is', age, 'years old')
         anne is 12 years old
         beth is 45 years old
         george is 32 years old
         damon is 102 years old
         ```
         2. zip中的序列1和序列2长度不同时, zip将在最短序列用完后停止组合
         ```
         >>> list(zip(range(5), range(100))
         [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
         ```
      + 迭代时获取索引
         1. enumerate(列表)可以返回索引及相应value
         ```
         for index, string in enumerate(strings):
            if 'xxx' in string:
               strings[index] = '[censored]'
         ```
      + 反向迭代和排序迭代
         1. reversed()原序列的反向迭代, 参数为序列或者可迭代对象, 返回一个新的可迭代对象, 不修改原对象
         ```
         >>> list(reversed('Hello, world！'))
         ['！', 'd', 'l', 'r', 'o', 'w', ' ', ',', 'o', 'l', 'l', 'e', 'H']
         >>> reversed('Hello, world!')
         <reversed object at 0x000001646485E748>
         ```
         2. sorted()将原序列排序迭代, 接受参数key为排序依据的函数, 默认是根据编码从小到大, 返回一个新列表, 不改变原序列
         ```
         >>> sorted([4, 3, 6, 8, 3])
         [3, 3, 4, 6, 8]
         >>> sorted([4, 3, 6, 8, 3], reverse = True)
         [8, 6, 4, 3, 3]
         >>> sorted('aBc')
         ['B', 'a', 'c']
         >>> sorted('aBc', key=str.lower)
         ['a', 'B', 'c']
         ```
      + 跳出循环
         1. break可以结束循环
         2. continue结束当前迭代, 直接跳到下一次迭代开始
         3. 可以使用永久循环`while True`, 然后在需要结束循环的时候使用`if/break`, 可以在任何条件下结束循环
         ```
         while True:
            word = input('Please enter a word:')
            if not word: break
            print('The word was ', word)
         ```
      + 循环的else语句
         1. while循环的else语句, 当循环变量不满足时执行else语句
         ```
         [初始化语句]
         while (循环变量):
	         语句块
	         [迭代语句]
         else:
	         语句块
         ```
         2. for循环的else语句, 当for循环完成所有扫描执行else语句
         ```
         for <变量> in <序列>:
	         循环体语句块
         else:
	         语句块
         ```
         3. 如果break结束循环, 则else语句不执行
      + 列表推导
         1. 语法格式
         ```
         variable = [out_exp_res for out_exp in input_list if out_exp == 2]
           # out_exp_res:　　列表生成元素表达式，可以是有返回值的函数。
           # for out_exp in input_list：　　迭代input_list将out_exp传入out_exp_res表达式中。
           # if out_exp == 2：　　根据条件过滤哪些值可以。
         ```
         ```
         >>> [x for x in range(31) if x % 3 == 0]
         [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
         # 30以内所有能被3整除的数
         >>> [(x, y) for x in range(3) for y in range(3)]
         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
         ```
         2. 嵌套列表推导
         ```
         >>> vec = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
         >>> [num for item in vec for num in item]
         [1, 2, 3, 4, 5, 6, 7, 8, 9]
         ```
      + 字典推导
         1. 使用大括号可以完成字典推导
         2. for语句前面有两个表达式, 用冒号分开, 分别代表key及value
         ```
         >>> squares = {i : "{} squared is {}.".format(i, i**2) for i in range(10)}
         >>> squares[8]
         '8 squared is 64.'
         ```
   + 其他语句
      + pass语句
        
         1. 什么都不做, 空语句
      + del语句
         1. 垃圾收集, 没有变量关联引用的内存, 无法获取或者使用, python解释器会直接将其删除
         2. del语句可以删除变量引用, 但是无法删除value本身
         3. 无用的value会被python解释器自动删除
         ```
         >>> x = ['Hello', 'world']
         >>> y = x
         >>> y[1] = 'python'
         >>> x
         ['Hello', 'python']
         >>> del x
         >>> y 
         ['Hello', 'python']
         ```
      + exec函数
         1. exec()将字符串作为代码执行
         2. 接受三个参数, 代码字符串, 全局变量命名空间为一个字典, 局部变量命名空间可以为任何映射
         3. 命名空间是为了存储执行代码字符串产生的变量
         ```
         >>> from math import sqrt
         >>> scope = {}
         >>> exec("sqrt = 1", scope)
         >>> sqrt(4)
         2.0
         >>> scope['sqrt']
         1
         >>> list(scope.keys())
         ['sqrt', '__builtins__']
         ```
         4. scope中包含了很多内置函数和值, __builtins__
      + eval语句
         1. 计算用字符串表示的python表达式的值, 并且返回结果
         ```
         >>> eval(input('Enter an expression: '))
         Enter an expression： 6 + 18*2
         42
         ```
         2. eval()也可以接受命名空间的参数
         3. 可以提前在命名空间中添加参数然后eval()使用
         ```
         >>> scope = {'x': 2, 'y': 3}
         >>> eval('x*y', scope)
         6
         ```
      
# Chapter 6 抽象 函数定义
   + 自定义函数
      + callable()函数可以判断某个对象是否可以调用
      ```
      >>> import math
      >>> x = 1
      >>> y = math.sqrt
      >>> callable(x)
      False
      >>> callable(y)
      True
      ```
      + def语句
         1. def 函数名(参数): block
         ```
         def Hello(name):
            return 'Hello, ' + name + '!'
         ```
         2. 函数的说明文档, 文档字符串放在def语句的后面, 函数的开头
         3. 可以通过help()函数或者函数名.__doc__查看说明文档
         ```
         def square(x):
            'Calculates the square of the number x'
            return x * x
         >>> square.__doc__
         'Calculates the square of the number x'
         >>> help(square)
         Help on function square in module __main__:
         square(x)
             Calculates the square of the number x
         ```
         4. 没有返回值的函数可以不加`return`或者`return`后面什么都不返回, 单独的`return`只是为了结束函数
   + 参数
      + 形参是字符串、数字或者元组时, 在函数内进行赋值操作, 不会影响原来函数外的变量
      + 形参传入是列表时, 在函数内修改列表的某一项, 函数外的原列表也会改变
      ```
      >>> def change(n):
      ···    n[0] = "Mr.Gumby"
      >>> names = ["Mr.Entity", "Mr.Thing"]
      >>> change(names)
      >>> names
      ['Mr.Gumby', 'Mr.Thing']
      ```
      + 关键字参数
         1. 可以使用名称指定参数, 不用考虑传入形参的顺序
         2. 关键字参数可以指定默认值, 从而调用函数时不用传入形参
         ```
         >>> def hello3(greeting="Hello", name='world'):
         ···    print("{}, {}!".format(greeting, world))
         >>> hello3()
         Hello, world!
         >>> hello3("Greetings")
         Greetings, world!
         >>> hello3("Greetings", "universe")
         Greetings, universe!
         >>> hello3(name='Gumby')
         Hello, Gumby!
         ```
      + 星号参数
         1. 可以同时收集多个参数, 将所有值都放在一个元组中
         ```
         >>> def print_params(*params):
         ···    print(params)
         >>> print_params(1, 2, 3)
         (1, 2, 3)
         ```
         2. 星号参数也可以放在中间, 后续的参数调用使用关键字
         ```
         >>> def in_the_middle(x, *y, z):
         ···    print(x, y, z)
         >>> in_the_middle(1, 2, 3, 4, 5, 6, z=7)
         1 (2, 3, 4, 5, 6) 7
         ```
         3. 星号参数不会收集关键字参数, ***要收集关键字参数可以使用两个星号, 得到的收集结果是一个字典***
         ```
         >>> def print_params_3(**params):
         ···    print(params)
         >>> print_params_3(x=1, y=2, z=3)
         {'x': 1, 'y': 2, 'z': 3}
         ```
      + 分配参数
         1. 星号参数也可以对传入的元组、字典等分配参数, 在调用函数时使用`*`
         2. `**`可以将字典中的value分配给关键字参数
         ```
         >>> def add(x, y):
         ···    return x+y
         >>> params = (1, 2)
         >>> add(*params)
         3
         >>> names = {"name": 'Sir Robin', 'greeting': 'Well met'}
         >>> hello3(**names)
         Well met, Sir Robin!
         ```
   + 作用域
      + 使用`globals()`可以返回全局变量的字典, 在函数内部访问同名的全局变量
      + 可以在函数内定义局部变量时使用`global`描述字, 关联到函数外的全局变量
      ```
      >>> def combine(params):
      ···    print(params+globals()["params"])
      >>> params = "berry"
      >>> combine('shrub')
      shrubberry
      >>> x = 1
      >>> def change_global():
      ···    global x
      ···    x = x + 1
      >>> change_global()
      >>> x
      2
      ```
   + 递归
     
      + 函数调用其自身, 需要有出口条件, 何时结束递归返回value
   + 几种特殊函数
      + map()
         1. map(函数名称, 序列), 对序列中所有元素执行函数
         ```
         >>> num = map(str, range(10))
         >>> list(num)
         ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
         >>> num
         <map object at 0x000002342420E248>
         ```
      + filter()
         1. filter(函数名称, 序列), 包含对序列执行函数得到结果为True的元素
         ```
         >>> def func(x):
         ···    return x.isalnum()
         >>> seq = ["foo", "x41", "?!", "***"]
         >>> res = filter(func, seq)
         >>> list(res)
         ['foo', 'x41']
         >>> res
         <filter object at 0x000002342447E508>
         ```
      + reduce()函数
        
         1. reduce(函数名称, 序列), 使用函数将序列的前两个元素合二为一, 然后结果与第三个元素合二为一, 以此类推
      + map()和filter()可以被列表推导代替
         ```
         >>> [str(x) for x in range(10)]
         ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
         >>> [x for x in seq if x.isalnum()]
         ['foo', 'x41']
         ```
      + lambda表达式
         1. 语法格式: `lambda 参数(可以多个) : 计算表达式`
         ```
         >>> a = lambda x, y:x+y
         >>> a(3, 6)
         9
         ```
         2. 常与map(), filter(), reduce()一起使用, 作为执行函数

# Chapter 7 抽象 自定义对象
   + 对象
      + 多态
         1. 同一个方法, 可以对多种不同类型的对象执行, 这就是多态
         2. 内置运算符和函数也存在多态, 例如`+`
      + 封装
         1. 向外部隐藏不必要的细节
      + 继承
         1. 创建新类型时可以在原有类型的基础上进行改进
   + 类
      + 子类与超类
         1. 子类的实例拥有超类的所有方法, 定义子类只需要定义多出来的方法
         2. 子类也可以重写超类中的方法
      + 自定义类
         1. 格式, `class 类名称: block`
         2. 在旧版本python中, 创建新式类要在开头加`__metaclass__ = type`
         3. 方法都包含参数`self`, 指向调用该方法的具体对象本身
         ```
         >>> class person:
         ···    def set_name(self, name):
         ···         self.name = name
         ···    def get_name(self):
         ···         return self.name
         ···    def greet(self):
         ···         print("Hello, world! I'm {}".format(self.name))
         >>> foo = person()
         >>> foo.set_name("wzy")
         >>> foo.greet()
         Hello, world! I'm wzy
         ```
      + 属性 方法 函数
         1. 方法和函数的区别在于`self`, 方法的第一个参数关联到它所属的实例
         2. ***私有方法或者属性, 让其名称以两个下划线打头即可***
         ```
         >>> class Secretive:
         ···    __name = "wzy"
         ···    def get_name(self):
         ···        return self.__name
         >>> x = Secretive()
         >>> x.get_name()
         "wzy"
         >>> x.__name
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
         AttributeError: 'Secretive' object has no attribute '__name'
         ```
         3. ***从类外访问私有属性或方法, 对属性或者方法名称进行转换, 在开头加一个下划线和类名即可***
         ```
         >>> x._Secretive__name
         "wzy"
         ```
         4. 以一个下划线开头的名称, 也有一定私有效果, `from module import *`不会导入以一个下划线开头的名称
      + 类命名空间
         1. 创建类时, 就会创建一个类的命名空间, 存储类的属性
         2. 属性包括
            1. 静态属性: 直接定义在类下面, 直接和类名关联的变量
            2. 对象属性: 在类内和self关联, 类外和对象关联的变量
            3. 动态属性: 方法
         3. 对象查找属性先在自己空间查找, 找不到再去类空间查找, 然后再去父类, 最后找不到则报错
         4. 静态属性是属于所有对象的, 如果发生修改则共享, 如果是赋值, 则是独立的, 只是修改的这个对象被赋值
         ```
         >>> class MemberCounter:
         ···     members = 0
         ···     def init(self):
         ···        MemberCounter.members += 1
         >>> m1 =MemberCounter()
         >>> m1.init()
         >>> MemberCounter.members
         1
         >>> m2 =MemberCounter()
         >>> m2.init()
         >>> MemberCounter.members
         2
         >>> m1.members
         2
         >>> m2.members
         2
         >>> m1.members = "Two"
         >>> m2.members
         2
         ```
      + 指定超类
         1. `class 类型(超类): block`
         2. 可以新定义的方式重写超类中的方法
         3. issubclass(类1, 类2), 判断类1是否为类2的子类
         4. 特殊属性`__bases__`可以得到一个类的基类
         5. isinstance(对象, 类), 判断对象是否为该类的一个实例
         6. 特殊属性`__class__`, 可以得到一个对象的类型
         ```
         >>> class Filter:
         ···     pass
         >>> class SPAM(Filter):
         ···     pass
         >>> issubclass(SPAM, Filter)
         True
         >>> issubclass(Filter, SPAM)
         False
         >>> SPAM.__bases__
         (<class '__main__.Filter'>,)
         >>> Filter.__bases__
         (<class 'object'>,)
         >>> x = SPAM()
         >>> x.__class__
         <class '__main__.SPAM'>
         >>> isinstance(x, SPAM)
         True
         >>> isinstance(x, Filter)
         True
         ```
      + 多重继承
         1. `class 类型(父类1, 父类2, 父类3)`
         2. ***如果父类中有方法重名, 则排名前面的父类的方法会将后面的父类的同名方法覆盖***
      + 接口
         1. python没有显示定义接口的方法, 可以通过一些函数或者属性来判断对象是否符合要求
         2. hasattr(对象, 属性), 判断对象是否包含属性
         3. getattr(对象, 属性, 默认值), 获取对象的某个属性, 如果不存在则返回默认值
         4. setattr(对象, 属性, value), 设置对象的属性=value
         5. 查看对象中存储的所有值, 调用`__dict__`属性, 只能查看对象属性
         ```
         >>> class test:
         ···    number = 1
         ···    def init(self):
         ···       self.name = "wzy"
         >>> x = test()
         >>> x.init()
         >>> x.__dict__
         {'name': "wzy"}
         ```
      + 抽象基类
         1. 通过引入模块abc, 可以定义抽象类, 规定子类必须实现的方法
         2. 抽象类不可以被实例化, 可以被继承
         3. `@abstractmethod`将方法标记为抽象方法, 子类中必须实现
         ```
         >>> from abc import ABC, abstractmethod
         >>> class Talker(ABC):
         ···    @abstractmethod
         ···    def talk(self):
         ···       pass
         
         ```
         4. 如果抽象基类的子类没有实现抽象方法, 则它也是抽象类, 是不能被实例化的
         ```
         >>> class knigget(Talker):
         ···    pass
         >>> k = knigget()
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
         TypeError: Can't instantiate abstract class knigget with abstract methods talk
         ```
         5. 较旧版本的python, 可以在class语句后面`__meteclass__=ABCMeta`, 或者`Talker(metaclass=ABCMeta)`
         6. `类1.register(类2)`可以将类2的所有对象看作是类1的对象, 可以通过isinstance()检查, 但是这样做抽象基类的保障就会损失
         
# Chapter 8 异常
   + 异常
      + 程序的错误状态, 异常对象没有处理的话程序将终止并显示错误信息
      + 异常的传播是由里向外, 层层向上的
   + 引发异常
      + `raise+异常对象`
      ```
      >>> raise Exception
      Traceback (most recent call last):
        File "<input>", line 1, in <module>
      Exception
      >>> raise Exception("overload")
      Traceback (most recent call last):
        File "<input>", line 1, in <module>
      Exception: overload
      ```
   + 自定义异常类
     
      + `class SomeCustomException(Exception): pass`, 像创建其他类一样, 必须直接或间接继承Exception
   + 捕获异常
      + 使用`try/except`语句
      ```
      >>> try:
      ···    x = int(input("first number"))
      ···    y = int(input("second number"))
      ···    print(x / y)
      ··· except ZeroDivisionError:
      ···    print("The second number can't be zero")
      >>> first number 5
      >>> second number 0
      The second number can't be zero
      ```
      + 捕获异常后, 可以使用`raise`使错误继续向上传播
      + 异常继续向上传播可以使用别的异常, `raise from`语句可以记录异常的上下文, 也可以使用None, 代表不记录原异常
      ```
      >>> try:
      ···    x = int(input("first number"))
      ···    y = int(input("second number"))
      ···    print(x / y)
      ··· except ZeroDivisionError:
      ···    print("The second number can't be zero")
      ···    raise ValueError from ZeroDivisionError
      >>> first number 5
      >>> second number 0
      The second number can't be zero
      ZeroDivisionError
      The above exception was the direct cause of the following exception:
      Traceback (most recent call last):
        File "<input>", line 7, in <module>
      ValueError
      ```
      ```
      >>> try:
      ···    x = int(input("first number"))
      ···    y = int(input("second number"))
      ···    print(x / y)
      ··· except ZeroDivisionError:
      ···    print("The second number can't be zero")
      ···    raise ValueError from None
      >>> first number 5
      >>> second number 0
      The second number can't be zero
      Traceback (most recent call last):
        File "<input>", line 7, in <module>
      ValueError
      ```
      + 捕获多个异常
         + 使用多个except语句
         + `except (异常1, 异常2, 异常3)`
      + 打印异常
      ```
      >>> try:
      ···    x = int(input("first number"))
      ···    y = int(input("second number"))
      ···    print(x / y)
      ··· except ZeroDivisionError as e:
      ···    print(e)
      >>> first number 5
      >>> second number 0
      division by zero
      ```
      + 捕获所有异常
         + `except: `可以捕获所有异常
         + 通常使用`except Exception as e: `仅捕获Exception的异常
      + 异常中的else语句
        
         + `try:pass except: pass else: pass`在没有异常捕获时执行else, 有异常时执行except
      + finally语句
         + 不管try语句发生什么异常, 都将执行finally语句
         ```
         >>> try:
         ···    pass
         ··· except:
         ···    pass
         ··· else:
         ···    pass
         ··· finally:
         ···    pass
         ```
   + 警告
      + 使用warnings模块中的warn函数可以发出警告, 警告只显示一次
      ```
      >>> from warnings import warn
      >>> warn("I've got a bad feeling about that")
      <input>:1: UserWarning: I've got a bad feeling about that
      ```
      + 使用warnings模块中的filterwarnings()可以过滤警告
         1. filterwarnings(action, category), action可以是ignore或者error, 即忽略或者升级为错误
         ```
         >>> from warnings import filterwarnings
         >>> filterwarnings("ignore")
         >>> warn("Anyone out there")
         >>> filterwarnings("error")
         >>> warn("Anyone out there")
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
         UserWarning: Anyone out there
         ```
         2. 可以根据category来过滤特定类型的警告
         ```
         >>> filterwarnings("error")
         >>> warn("Anyone out there", DeprecationWarning)
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
         DeprecationWarning: Anyone out there
         >>> filterwarnings("ignore", category=DeprecationWarning)
         >>> warn("Anyone out there", DeprecationWarning)
         >>> warn("Anyone out there")
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
         UserWarning: Anyone out there
         ```
      
# Chapter 9 特殊方法 属性 迭代器
   + 新式类与旧式类
      + 新式类直接或者间接的继承object, python3中都是新式类
      + 旧式类主要在python2中, 在文件开头`__metaclass__=type`添加, 则变为新式类
   + 构造函数
      + `__init__`方法, 在对象被创建时自动调用
      + 销毁函数, `__del__`在对象被销毁之前调用, ***不建议使用, 因为不知道准确的销毁时间***
      ```
      >>> class FooBar:
      ···    def __init__(self, value=42):
      ···       self.somevar = value
      >>> x = FooBar("test")
      >>> x.somevar
      'test'
      ```
   + 继承时重写特殊构造函数
      + 子类重写构造函数时, 需要调用父类的构造函数, 否则无法使用父类的一些属性或者方法
      + 针对旧式类, 在构造函数中添加`父类.__init__(self), 即可调用父类的构造函数`
      ```
      >>> class Bird:
      ···    def __init__(self):
      ···        self.hungry = True
      ···    def eat(self):
      ···        if self.hungry:
      ···            print("Aaaah...")
      ···            self.hungry = False
      ···        else:
      ···            print("No, thanks")
      >>> class SongBird(Bird):
      ···    def __init__(self):
      ···        Bird.__init__(self)
      ···        self.sound = "Squawk"
      ···    def sing(self):
      ···        print(self.sound)
      >>> sb = SongBird()
      >>> sb.sing()
      Squawk
      >>> sb.eat()
      Aaaah...
      >>> sb.eat()
      No, thanks
      ```
      + 针对新式类, 可以使用super(), 在构造函数中添加`super().__init__()`或者`super(子类, self).__init__()`
      ```
      >>> class SongBird(Bird):
      ···    def __init__(self):
      ···        super().__init__()
      ···        self.sound = "Squawk"
      ···    def sing(self):
      ···        print(self.sound)
      >>> sb = SongBird()
      >>> sb.sing()
      Squawk
      >>> sb.eat()
      Aaaah...
      >>> sb.eat()
      No, thanks
      ```
   + 基本的序列和映射协议
      + 协议有点类似于接口, 指定应该实现哪些方法以及这些方法应该做什么
      + 不可变对象需要实现2个方法, 可变对象需要实现4个
         1. __len__(self)
            + 返回集合包含的项数, 对序列是元素个数, 对映射是键值对数
            + 如果返回0, 则对象在布尔测试中返回False
         2. __getitem__(self, key)
            + 返回与指定键相关联的值, 对序列key是0~n-1的整数或者负数, 对映射key可以是任意类型
         3. __setitem__(self, key, value)
            + 设置相关联的键值对
            + 仅在可变对象中实现
         4. __delitem__(self, key)
            + 删除与key相关联的值
            + 仅在可变对象中实现
      + 额外要求
         + 对于序列, x[-n] = x[len(x)-n]等效
         + 如果key的类型不合适, 会引发TypeError
         + 如果索引超出范围会引发IndexError
      + 从list, dict， str继承
         + 序列还有很多其他特殊方法, 简单的方法是从list, dict, str继承
         ```
         >>> class counterList(list):
         ···    def __init__(self, *args):
         ···        super().__init__(*args)
         ···        self.counter = 0
         ···    def __getitem__(self, index):
         ···        self.counter += 1
         ···        return super().__getitem__(index)
         >>> cl = counterList(range(10))
         >>> cl[5]
         5
         ```
         + 更多的魔法方法
            + [python手册链接](https://docs.python.org/zh-cn/3/reference/datamodel.html#special-method-names)
   + 特性
      + 通过存取方法定义的属性叫做特性
      + `特性名称 = property(获取方法, 设置方法)`, 这样就可以用`对象.特性`来使用, 背后隐藏了存取方法
      ```
      >>> class Rectangle:
      ···      def __init__(self):
      ···        self.width = 0
      ···        self.height = 0
      ···      def get_size(self):
      ···        return self.width, self.height
      ···      def set_size(self, size):
      ···        self.width, self.height = size
      >>> class NewRectangle:
      ···      def __init__(self):
      ···        self.width = 0
      ···        self.height = 0
      ···      def get_size(self):
      ···        return self.width, self.height
      ···      def set_size(self, size):
      ···        self.width, self.height = size
      ···      size = property(get_size, set_size)
      >>> r =Rectangle()
      >>> r.width = 10
      >>> r.height = 15
      >>> r.get_size()
      (10, 15)
      >>> r.set_size((150, 100))
      >>> r.width
      150
      >>> nr = NewRectangle()
      >>> nr.width = 10
      >>> nr.height = 15
      >>> nr.size
      (10, 15)
      >>> nr.size = 150, 100
      >>> nr.width
      150
      ```
      + property()可以有四个参数
         1. 如果没有指定参数, 则特性不可读也不可取
         2. 指定一个参数, 必须是获取方法, 特性只可读
         3. 第三个参数是删除方法, 可选
         4. 第四个参数是可选, 文档字符串, 关键字参数为`fget, fset, fdel, doc`
   + 装饰器
      + 闭包
         1. 在函数内部定义函数, 并将内部函数作为外部函数的返回值
         2. 内部函数可以引用外部函数的参数和局部变量，当外部函数返回内部函数时，相关参数和变量都保存在返回的函数中
         ```
         >>> def delay_sum(*args):
         ···    def sum():
         ···        res = 0
         ···        for x in args:
         ···            res += x
         ···        return res
         ···    return sum
         >>> f = delay_sum(1, 3, 5, 7, 9)
         >>> f
         >>> <function delay_sum.<locals>.sum at 0x000001A2BDE3C4C8>
         >>> f()
         25
         ```
         3. 当我们调用外部函数时，每次调用都会返回一个新的函数，即使传入相同的参数
         ```
         >>> f1 = delay_sum(1, 3, 5, 7, 9)
         >>> f2 = delay_sum(1, 3, 5, 7, 9)
         >>> f1 is f2
         False
         ```
         4. ***闭包返回的函数并没有立刻执行，而是直到调用了f()才执行***
      + 装饰器
         1. 在代码运行期间, 动态增加功能的方式, 简而言之, 就是拓展原来函数功能的函数
         2. 本质上讲, 装饰器就是一个闭包
         ```
         >>> def now():
         ···     print("2020-02-20")
         >>> now.__name__
         'now'
         >>> def log(func):
         ···     def wrapper(*args, **kw):
         ···         print("call {}".format(func.__name__))
         ···         return func(*args, **kw)
         ···     return wrapper
         >>> @log
         ··· def now():
         ···     print("2020-02-20")
         >>> now.__name__
         'wrapper'
         >>> now()
         call now
         2020-02-20
         ```
         3. log函数是一个装饰器, 通过在函数定义之前使用`@函数名`来调用, 上述例子中`@log def now: `相当于执行了`now = log(now)`, 此时now指向装饰器中返回的函数wrapper, 而wrapper函数在调用原函数之前又进行了修饰
         4. 如果装饰器本身需要传入函数, 需要编写一个返回装饰器的高阶函数
         ```
         >>> def log(text):
         ···     def decorate(func):
         ···         def wrapper(*args, **kw):
         ···             print("{} {}".format(text, func.__name__))
         ···             return func(*args, **kw)
         ···         return wrapper
         ···     return decorate
         >>> @log("execute")
         ··· def now():
         ···     print("2020-02-20")
         >>> now()
         execute now
         2020-02-20
         ```
         5. 上述是一个三层嵌套, 效果如`now = log("execute")(now)`
   + 静态方法和类方法
      + python中包括三种方法: 实例方法, 类方法, 静态方法
         1. 实例方法即之前提到包含参数self的方法, 必须要有实例化对象才可以调用, 即调用实例方法需要类的实例(对象)
         2. 类方法需要包含类似self的参数, 一般定义为cls, 关联到类本身, 可以调用类的属性和方法
         3. 静态方法不需要self或者cls参数, 调用时也不需要实例对象, 类可以直接调用
      + 定义
         1. 静态方法定义时需要使用`@staticmethod`装饰器
         2. 类方法定义时需要使用`@classmethod`装饰器
      + 调用
         1. 实例定义的变量只能被实例访问, 静态方法和类方法无法访问
         2. 直接在类中定义的静态变量既可以被实例方法访问, 也可以被静态方法和类方法访问
         3. 静态方法和类方法中不能使用实例方法, 实例方法中可以使用静态方法和类方法
         4. 实例方法只有实例对象可以调用, 静态方法和类方法既可以实例对象调用也可以类名直接调用
         ```
         >>> class Myclass:
         ···    # 定义一个静态变量，可以被静态方法和类方法访问
         ···    name = 'Bill'
         ···    def __init__(self):
         ···         print('Myclass的构造方法被调用')
         ···         # 定义实例变量，静态方法和类方法不能访问该变量
         ···         self.value = 20
         ···     # 定义静态方法
         ···     @staticmethod
         ···     def run():
         ···         # 访问Myclass类中的静态变量name
         ···         print('*', Myclass.name, '*')
         ···         print('Myclass的静态方法run被调用')
         ···     # 定义类方法
         ···     @classmethod
         ···     # 这里cls是类的元数据，不是类的实例
         ···     def do(cls):
         ···         print(cls)
         ···         # 访问Myclass类中的静态变量name
         ···         print('[', cls.name, ']')
         ···         print('调用静态方法run')
         ···         cls.run()
         ···         # 在类方法中不能访问实例变量，否则会抛出异常（因为实例变量需要用类的实例访问） print(cls.value)
         ···         print('成员方法do被调用')
         ···     # 定义实例方法
         ···     def do1(self):
         ···         print(self.value)
         ···         print('<', self.name, '>')
         ···         print(self)
         >>> # 调用静态方法run
         >>> Myclass.run()
         * Bill *
         Myclass的静态方法run被调用
         >>> # 创建Myclass类的实例
         >>> c = Myclass()
         Myclass的构造方法被调用
         >>> # 通过类的实例也可以调用类方法
         >>> c.do()
         <class '__main__.Myclass'>
         [ Bill ]
         调用静态方法run
         * Bill *
         Myclass的静态方法run被调用
         成员方法do被调用
         >>> # 通过类访问类的静态变量
         >>> print('Myclass2.name', '=', Myclass.name)
         Myclass2.name = Bill
         >>> # 通过类调用类方法
         >>> Myclass.do()
         <class '__main__.Myclass'>
         [ Bill ]
         调用静态方法run
         * Bill *
         Myclass的静态方法run被调用
         成员方法do被调用
         >>> # 通过类的实例访问实例方法
         >>> c.do1()
         20
         < Bill >
         <__main__.Myclass object at 0x000001A2BDFCAA48>
         ```
   + 调用属性时的特殊方法
      1. __getattribute__(self, name), 在属性被访问时自动调用
      2. __getattr__(self, name), 在属性被访问而对象没有这样的属性时被自动调用
      3. __setattr__(self, name), 在给属性赋值时被自动调用
      4. __delattr__(self, name), 试图删除属性时被自动调用
   + 迭代器
      + 可迭代对象`Iterable`, 必须实现方法`__iter__()`, 可以用于for循环, 可以通过`isinstance(对象, Iterable)`来判断是否为可迭代对象
         1. `__iter__()`返回结果是一个迭代器
      + 迭代器`Iterator`, 必须实现方法`__next__()`返回下一个值, 可以通过`isinstance(对象, Iterator)`来判断是否为迭代器
         1. 迭代器也可以被内置函数next()调用, `next(迭代器)`也能返回下一个值
      + list, dict, str不是迭代器, 但是可以通过iter()变为迭代器
      ```
      >>> from collections import Iterable, Iterator
      >>> a = [1, 2, 3, 4]
      >>> isinstance(a, Iterable)
      True
      >>> isinstance(a, Iterator)
      False
      >>> isinstance(iter(a), Iterator)
      ```
      + 迭代器对象表示的是一个数据流，迭代器对象可以被next()函数调用并不断返回下一个数据，直到没有数据时抛出StopIteration错误。可以把这个数据流看做是一个有序序列，但我们却不能提前知道序列的长度，只能不断通过next()函数实现按需计算下一个数据，所以迭代器的计算是惰性的，只有在需要返回下一个数据时它才会计算。
      + 迭代器甚至可以表示一个无限大的数据流, 例如全体自然数
   + 生成器
      + 包含`yield`语句的函数叫做生成器
         1. 生成器不是使用`return`返回一个值, 而是生成多个值, 每次使用`yield`生成一个值后, 函数将冻结, 即在此停止执行, 等待被重新唤醒, 然后再从停止的地方开始继续执行
         2. 生成器是迭代器对象, 可以用于for循环
         3. 生成器代码遇到`yield`意味生成一个值, 而`return`意味着生成器应停止执行
         ```
         >>> nested = [[1, 2], [3, 4], [5]]
         >>> def flatten(x):
         ···     for y in x:
         ···         for element in y:
         ···             yield element
         >>> for num in flatten(nested):
         ···     print(num)
         1
         2
         3
         4
         5
         >>> isinstance(flatten(nested), Iterator)
         True
         ```
      + 生成器推导
         + 类似于列表表达式, 但是使用`(生成器表达式)`圆括号括起来
         + 生成器表达式不同于列表表达式, 不会立即实例化所有元素, 每迭代一次计算一次
         ```
         >>> g = ((i+2) ** 2 for i in range(2, 27))
         >>> next(g)
         16
         >>> next(g)
         25
         ```
      + 递归式生成器
         ```
         >>> def flatten(x):
         ···     try:
         ···         for y in x:
         ···             for element in flatten(y):
         ···                 yield element
         ···     except TypeError:
         ···         yield x
         ```
         + 递归式生成器不可以递归字符串, 因为字符串的第一个元素又是一个长度为1的字符串, 而长度为1的字符串的第一个元素又是其本身, 会导致无穷迭代
         + 对于字符串需要在生成器开头进行检查, ***最简单的方式是检查其是否拥有类似字符串的行为, 而不是直接进行类型检查***
         ```
         >>> def flatten(x):
         ···     try:
         ···         try:x+""
         ···         except: pass
         ···         else: raise TypeError
         ···         for y in x:
         ···             for element in flatten(y):
         ···                 yield element
         ···     except TypeError:
         ···         yield x
         >>> list(flatten(['foo', ['bar', ['baz']]]))
         ['foo', 'bar', 'baz']
         ```
      + 生成器 send方法
         1. send方法, 可以与生成器进行交互, 该方法有一个参数，***该参数指定的是上一次被挂起的yield语句的返回值***
         2. 生成器使用next()时, yield语句的返回值为None
         3. send方法和next方法唯一的区别是在执行send方法会首先把上一次挂起的yield语句的返回值通过参数设定，从而实现与生成器的交互。但是需要注意，在一个生成器对象没有执行next方法之前，由于没有yield语句被挂起，所以执行send方法会报错
         4. 也可以直接执行`send(None)`, 等价于next()
         ```
         >>> def repeater(value):
         ···     while True:
         ···         new = (yield value)
         ···         if new is not None: value = new
         >>> r = repeater(42)
         >>> next(r)
         42
         >>> next(r)
         42
         >>> r.send("wzy")
         "wzy"
         >>> next(r)
         "wzy"
         ```
      + 生成器 throw方法
         1. throw方法用于再生成器中挂起的yield语句处抛出一个异常, 如果异常没有被捕获, 则报错生成器终止, 如果异常被处理, 则程序继续运行到下一句yield处
         ```
         >>> def test():
         ···     value = 1
         ···     while True:
         ···         yield value
         ···         value += 1
         >>> g = test()
         >>> next(g)
         1
         >>> next(g)
         2
         >>> g.throw(Exception, "Error")
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
           File "<input>", line 4, in test
         Exception: Error
         ```
         ```
         >>> def test():
         ···     value = 1
         ···     while True:
         ···         try:            
         ···             yield value
         ···             value += 1
         ···         except:
         ···             value = 1
         >>> g = test()
         >>> next(g)
         1
         >>> next(g)
         2
         >>> next(g)
         3
         >>> g.throw(Exception, "Error")
         1
         >>> next(g)
         2
         >>> next(g)
         3
         ```
      + 生成器 close方法
         1. 生成器对象的close方法会在生成器对象方法的挂起处抛出一个GeneratorExit异常。GeneratorExit异常产生后，系统会继续把生成器对象方法后续的代码执行完毕
         ```
         >>> def myGenerator():  
         ···     try:
         ···         yield 1
         ···         yield 2
         ···     except GeneratorExit:
         ···         print ("aa")
         >>> gen = myGenerator()
         >>> next(gen)
         1
         >>> gen.close()
         aa
         ```
         2. GeneratorExit异常的产生意味着生成器对象的生命周期已经结束。因此，一旦产生了GeneratorExit异常，生成器方法后续执行的语句中，不能再有yield语句，否则会产生RuntimeError
         ```
         >>> def myGenerator():  
         ···     try:
         ···         yield 1
         ···         yield 2
         ···     except GeneratorExit:
         ···         print ("aa")
         ···     yield 3
         >>> gen = myGenerator()
         >>> next(gen)
         1
         >>> gen.close()
         aa
         Traceback (most recent call last):
           File "<input>", line 1, in <module>
         RuntimeError: generator ignored GeneratorExit
         ```

# Chapter 10 模块
   + 模块 
      + 添加路径
         1. 自定义模块添加路径
         ```
         >>> import sys
         >>> sys.path.append(自定义模块的路径)
         ```
         2. 查找解释器的默认搜索路径, ***一般会放在site-packages文件夹下, 这是默认的模块存储的地方***
         ```
         >>> import sys, pprint
         >>> pprint.pprint(sys.path)
         ['D:\\Program Files\\JetBrains\\PyCharm Community Edition '
          '2019.2.3\\helpers\\pydev',
          'D:\\Program Files\\JetBrains\\PyCharm Community Edition '
          '2019.2.3\\helpers\\third_party\\thriftpy',
          'D:\\Program Files\\JetBrains\\PyCharm Community Edition '
          '2019.2.3\\helpers\\pydev',
          'D:\\Program Files (x86)\\Python3\\python37.zip',
          'D:\\Program Files (x86)\\Python3\\DLLs',
          'D:\\Program Files (x86)\\Python3\\lib',
          'D:\\Program Files (x86)\\Python3',
          'C:\\Users\\lenovo\\AppData\\Roaming\\Python\\Python37\\site-packages',
          'D:\\Program Files (x86)\\Python3\\lib\\site-packages',
          'C:\\Users\\lenovo\\Desktop\\learn_python',
          'C:/Users/lenovo/Desktop/learn_python']
         ```
      + 导入模块时, 模块中的代码只执行一次, 因此多次导入和导入一次效果相同, 如果要重新加载, 可以使用importlib中的reload
      ```
      >>> import importlib
      >>> Hello = importlib.reload(hello)
      ```
      + 模块有自己的作用域, 意味着在模块中定义的类和函数以及赋值的变量都将成为模块的属性
      + 测试模块
         1. 在主程序和解释器的交互式提示符中, `__name__`的值是`__main__`, 而在导入的模块中这个变量被设置为该模块的名称
         2. 通过`if __name__ == '__main__'`来判断是否运行测试函数
   + 包
      + 包是一个文件夹目录, 底下可以包含其他模块, 模块一般是一些.py文件
      + 包的目录下必须包含`__init__.py`文件, 如果`import packages`, 即导入`__init__.py`文件的内容
      ```
      >>> import drawing   # 导入drawing包__init__.py
      >>> import drawing.colors  # 导入drawing包中的模块colors
      >>> from drawing import shapes # 导入模块shapes
      ```
   + 学习使用模块
      + 函数dir(), 列出对象的所有属性, 即模块的所有函数, 类, 变量
      ```
      >>> import copy
      >>> [n for n in dir(copy) if not n.startswith("_")]
      ['Error', 'copy', 'deepcopy', 'dispatch_table', 'error']
      ```
      + 变量`__all__`， 告诉解释器, 在`from copy import *`时, 应该导入的函数或者变量
      ```
      >>> copy.__all__
      ['Error', 'copy', 'deepcopy']
      ```
      + help函数, 帮助获取有关函数的说明文档和信息
      ```
      >>> help(copy.copy)
      Help on function copy in module copy:
      copy(x)
          Shallow copy operation on arbitrary Python objects.  
          See the module's __doc__ string for more info.
      ```
      + `__doc__`文档字符串
      + 查看源代码, `__file__`返回源代码的路径
      ```
      >>> copy.__file__
      'D:\\Program Files (x86)\\Python3\\lib\\copy.py'
      ```
   + sys模块
      + 与python解释器相关的变量和函数
      1. sys.argv包含传递给python解释器参数, 其中包括脚本名
      2. sys.exit退出当前程序, 可以提供一个整数或者字符串当作提示消息
      3. sys.modules是一个映射, 将模块名称映射模块
      4. sys.path是一个列表, 包含python解释器的查找模块路径
      5. sys.platform返回操作系统名称
      6. sys.stdin, sys.stdout, sys.stderr标准输入流、输出流和错误流
   + os模块
      + 访问有关操作系统的服务
      1. os.path包含查看, 创建, 删除目录及文件的函数, 以及一些操作路径的函数
      2. os.environ是一个映射, 包含系统环境变量
      3. os.system(command), 在子shell中执行操作系统命令
      4. os.sep, 是用于路径名中的分隔符
      5. os.pathsep, 多条路径之间的分隔符
      6. os.linesep, 文本文件中的行分隔符
   + fileinput模块
      1. fileinput.input(), 返回一个可以在for循环当中进行迭代的对象, 可以接受一个文件列表确定要迭代哪些文件, inplace参数设置为True则对文件就地进行处理, 输出内容写回到当前输入文件中, 此时可选参数backup用于给原始文件创建的备份文件指定扩展名
      2. fileinput.filename, 返回当前处理行所属文件的文件名
      3. fileinput.lineno, 返回当前行编号, 是累计值, 处理完一个文件时不会重置行号
      4. fileinput.filelineno, 返回当前行在当前文件中的行号
      5. fileinput.isfirstline(), 当前行为当前文件的第一行则返回True, 否则false
      6. fileinput.isstdin(), 当前文件为sys.stdin时则返回True, 否则false
      7. fileinput.nextfile(), 关闭当前文件并且移到下一个文件, 计数时候忽略跳过的行
      8. fileinput.close(), 关闭整个文件链并结束迭代
   + 集合模块 set
      1. 直接调用内置类set即可, 创建可以使用set()或者{}显示指定, 但是空集合不能`{}`, 这个默认为空字典
      ```
      >>> set(range(10))
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
      ```
      2. 集合主要用于成员资格检查, 因此不能包含重复元素
      ```
      >>> {1, 2, 3, 4, 4, 5, 3}
      {1, 2, 3, 4, 5}
      ```
      3. 可以执行标准集合操作, 交补并差
      4. 可以使用add或者remove函数, 添加或者删除元素
      5. 集合是可变的, 但是集合中的元素是不可变的, 因此集合中不能包含其他集合
      6. 可以使用frozenset类型, 表示不可变的集合
      ```
      >>> a =set()
      >>> b =set()
      >>> a.add(b)
      Traceback (most recent call last):
        File "<input>", line 1, in <module>
      TypeError: unhashable type: 'set'
      >>> a.add(frozenset(b))
      ```
   + 堆模块 heapq
      1. 堆是一种优先队列, 让你能够以任意顺序添加对象, 并随时找出最小元素
      2. 没有单独的堆类型, 使用列表来表示堆对象
      3. 堆特征, 堆对象中的元素排列没有严格规则, 但是位置i处的元素总是大于位置i//2处的元素, 反过来小于位置2i与2i+1处的元素, ***第一个元素总是最小元素***
      4. 常用函数

      | 函数 | 描述 |
      | --- | --- |
      | heappush(heap, x)  | 将x压入堆heap中 |
      | heappop(heap) | 从堆中弹出最小的元素 |
      | heapify(heap) | 让列表具备堆特征 |
      | heapreplace(heap, x) | 弹出最小的元素, 并将x压入heap中 |
      | nlargest(n, iter) | 返回iter中n个最大的元素 |
      | nsmallest(n, iter) | 返回iter中n个最小的元素 |
      
      ```
      >>> from heapq import *
      >>> from random import shuffle
      >>> data = list(range(10))
      >>> shuffle(data)
      >>> heap = []
      >>> for n in data:
      ···     heappush(heap, n)
      >>> heap
      [0, 1, 3, 2, 5, 7, 6, 9, 4, 8]
      >>> data 
      [1, 8, 7, 9, 4, 3, 6, 0, 2, 5]
      >>> heappop(heap)
      0
      >>> heap
      [1, 2, 3, 4, 5, 7, 6, 9, 8]
      ```
      5. 如果列表不是通过heappush创建的, 则在使用其他函数之前应该先heapify使其拥有堆特征
   + 双端队列
      1. deque类型, 包含在模块collections中
      2. 双端队列也需要从可迭代对象建立
      ```
      >>> from collections import deque
      >>> q = deque(range(5))
      >>> q
      deque([0, 1, 2, 3, 4])
      ```
      3. append(), appendleft(),可以从队列的首端或者末尾添加元素
      ```
      >>> q.append(5)
      >>> q.appendleft(6)
      >>> q
      deque([6, 0, 1, 2, 3, 4, 5])
      ```
      4. pop(), popleft()可以从队列的首端或者末尾弹出元素
      ```
      >>> q.pop()
      5
      >>> q.popleft()
      6
      >>> q
      deque([0, 1, 2, 3, 4])
      ```
      5. rotate()可以根据参数的正负将队列向左移或者向右移
      ```
      >>> q.rotate(1)
      >>> q
      deque([4, 0, 1, 2, 3])
      >>> q.rotate(-1)
      >>> q
      deque([0, 1, 2, 3, 4])
      ```
      6. extend()， extendleft(), 一次添加多个可迭代对象到队列末尾或者首端
   + time模块
      + 获取当前时间, 操作时间和日期, 从字符串中读取日期, 将日期格式化为字符串
      + 日期元组, (年, 月, 日, 小时0~23, 分, 秒, 星期0~6(0是星期一), 一年中第几天0~366, 夏令时(-1, 0, 1), )
      1. asctime(), 将当前时间转化为字符串, 也可以提供一个日期元组
      ```
      >>> import time
      >>> time.asctime()
      'Fri Feb 21 19:10:20 2020'
      ```
      2. localtime()可以将一个实数(从新纪元开始之后的秒数)转换为日期元组
      ```
      >>> time.localtime()
      time.struct_time(tm_year=2020, tm_mon=2, tm_mday=21, tm_hour=22, tm_min=50, tm_sec=54, tm_wday=4, tm_yday=52, tm_isdst=0)
      >>> time.localtime(0)
      time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=8, tm_min=0, tm_sec=0, tm_wday=3, tm_yday=1, tm_isdst=0)
      ```
      3. mktime()可以将日期元组转换为新纪元之后的秒数, 与localtime()功能相反
      4. sleep(sec)函数让解释器等待秒数,
      5. strptime(字符串)将一个类似asctime()函数返回的日期字符串变为日期元组
      6. time()返回现在国际标准时间, 以从新纪元开始的秒数
   + random模块
      + 生成伪随机数的函数
      1. random(), 返回一个0-1的随机实数, 包含1但是不包含0
      2. getrandbits(n), 返回一个整数, 其大小为指定位数n的随机二进制数
      ```
      >>> random.getrandbits(5)
      25
      >>> random.getrandbits(5)
      30
      ```
      3. uniform(a, b), 返回一个a~b(包含)的随机实数
      4. `randrange([start], stop, [step])`, 从`range(start, stop, step)`中随机选择一个数
      5. choice(seq), 从序列seq中随机选取一个元素
      6. shuffle(seq), 随机的打乱可变序列中的元素
      7. sample(seq, n), 从序列seq中随机地选择n个值不同的元素
   + shelve模块和json模块
   + re模块
      + 提供有关正则表达式的函数
         1. 正则表达式, 匹配文本片段的公式
         2. 通配符, `.`与除换行符之外的任何字符都匹配, 但是只匹配一个字符, 不与零个或者两个字符匹配
         3. 对特殊字符进行转义, 例如对句点, 前面添加反斜杠进行匹配
         4. 用方括号将字符串括起来创建字符集, 字符集与其包含的其中一个字符匹配
         5. 字符集可以表示范围`[a-z], [a-zA-Z0-9]`
         6. 排除指定的字符集, `[^abc]`表示除a, b, c之外的任何字符
         7. 二选一的特殊字符, `python | perl`表示python或者perl字符
         8. 圆括号括起来的子模式, `p(ython | erl)`, 表示圆括号中的字符二选一
         9. 在子模式后加`?`表示括号中的内容可包含一次或者不包含, `(www\.)python\.org`
         10. `(pattern)*`表示pattern可重复1次, 0次或者多次
         11. `(pattern)+`表示pattern可重复1次或者多次
         12. `(pattern){m, n}`表示pattern可重复m~n次
         13. 字符串的开头匹配, 使用`^`, 例如`^ht+p`与http://python.org匹配
         14. 字符串的结尾匹配, 使用`$`
      + re模块的函数
         1. compile(pattern), 将字符串表示的正则表达式转换为模式对象, re中的匹配函数都是使用模式对象进行查找, 匹配函数也可以传入正则表达式, 但是也会进行转换, 每次转换会降低效率
         2. re.search(pattern, string), pattern.search(string) 在给定字符串string中查找第一个与模式对象匹配的子串, 返回结果MatchObject真假 
         3. re.match是尝试在开头查找匹配的子串
         4. re.split()根据模式匹配的子串来分割字符串, 关键字参数maxsplit表示最多分割多少次
         ```
         >>> import re
         >>> text = 'alpha, beta,,,,,gamma     delta'
         >>> re.split('[, ]+', text)
         ['alpha', 'beta', 'gamma', 'delta']
         >>> re.split('[, ]+', text, maxsplit = 1)
         ['alpha', 'beta,,,,,gamma     delta']
         ```
         5. re.findall(), 返回一个列表, 其中包含所有与给定模式匹配的子串
         ```
         >>> pattern = re.compile("[a-zA-Z]+")
         >>> text = "Hm...., Err....are you sure? he said, sounding insecure"
         >>> pattern.findall(text)
         ['Hm', 'Err', 'are', 'you', 'sure', 'he', 'said', 'sounding', 'insecure']
         ```
         6. re.escape(), 是一个工具函数, 对字符串中所有可能被视为正则表达式运算符的字符进行转义
         ```
         >>> re.escape("www.python.org")
         'www\\.python\\.org'
         ```
         7. re.sub(pat, repl, string), 将字符串中与模式pat相匹配的子串替换为repl    
         ```
         >>> re.sub("\*([^\*]+)\*", r"<em>\1</em>", "Hello, *world*")
         'Hello, <em>world</em>'
         ```
      + MatchObject对象
         1. 查找与模式匹配的函数的返回结果
         2. 包含字符串中与模式相匹配的子串的信息, 相匹配的子串叫做group, 编组
         3. 编组就是放在圆括号内的子模式, 根据左边的括号数编号, 与模式pattern中的圆括号有关
         ```
         'There (was a (wee) (cooper)) who (lived in Fyfe)'
         group 0 There was a wee cooper who lived in Fyfe
         group 1 was a wee cooper
         group 2 wee
         group 3 cooper
         group 4 lived in fyfe
         ```
         4. group(编号), 返回给定模式中编组匹配的子串, 默认为0
         5. start(编号), 指定编组的子串的起始索引
         6. end(编号), 指定编组的子串的结束索引+1
         7. span(编号), 返回一个元组, 指定编组的子串的开始和结束索引+1
         ```
         >>> m = re.match(r"www\.(.*)\..{3}", "www.python.org")
         >>> m.group(1)
         'python'
         >>> m.start(1)
         4
         >>> m.end(1)
         10
         >>> m.span(1)
         (4, 10)
         ```
      

# Chapter 11 文件
   + open(文件名, mode), 打开文件
      + mode模式
      
      | 值 | 描述 |
      |---|---|
      | 'r' | 读取模式, 是默认值 |
      | 'w' | 写入模式, 文件不存在时创建文件, 写入模式下打开文件已有内容将删除 |
      | 'x' | 独占写入模式, 如果文件已存在则引发FileExistsError异常 |
      | 'a' | 附加模式, 在既有文件末尾继续写入 |
      | 't' | 文本模式, 默认值, 与其他模式结合使用 |
      | 'b' | 二进制模式, 与其他模式结合使用 |
      | '+' | 读写模式, 与其他模式结合使用, 表示既可读取也可写入 |
      
      + 'r+', 'w+'区别, 前者不会删除已有内容, 后者会删除已有内容
      + 关键字参数`newline`设置相应的行尾字符, python默认使用通用换行模式(\r, \n, \r\n)
   + 读取和写入
      1. 打开文件对象, 然后使用read()或者write()方法
      2. 在文本模式和二进制模式下, 分别将str和bytes作为数据
      3. file.write(写入内容), 返回写入内容长度
      4. file.read(读取内容长度), 默认是全部读取
      ```
      >>> f = open("somefile.txt")
      >>> f.write("Hello, ")
      7
      >>> f.write("world!")
      6
      >>> f.close()
      ```
      ```
      >>> f = open("somefile.txt")
      >>> f.read(4)
      "Hell"
      >>> f.read()
      "o, world!"
      ```
   + 在bash等shell中可以使用`|`连接多个命令, 并且前一个命令的标准输出就是后一个命令的标准输入
   ```
   $ cat somefile.txt | python somecript.py | sort
   ```
   + tell(), 返回当前位于文件什么位置
   ```
   >>> f = open("somefile.txt")
   >>> f.read(3)
   "012"
   >>> f.read(2)
   "34"
   >>> f.tell()
   5
   ```
   + seek(offset[, whence]), 将当前位置移到offset和whence指定的地方, offset表示偏移量字节数, whence默认io.SEEK_SET(0)即文档开头, 也可以设置为io.SEEK_CUR(1), 表示相对于当前位置, 或者io.SEEK_END(2)相对于文件末尾进行移动
   + 读取写入行
      + readline(), 读取文件的一行, 也可以提供非负参数, 表示最多读取几个字节内容
      + realines(), 读取文件中的所有行, 并返回一个列表
      + writelines(), 接受一个字符串序列, 写入到文件中, ***写入时候不会自动添加换行符, 必须自行添加***
   + 关闭文件
      + with语句, 打开文件并赋值给一个变量, 然后执行block, 结束block时自动关闭文件, 出现异常也可以直接关闭文件
      ```
      with open("somefile.txt") as somefile:
          do_something(somefile)
      ```
   + 迭代文件
      + 每次一个字符
      ```
      with open("somefile.txt") as f:
          while True:
              char = f.read(1)
              if not char: 
                  break
              process(char) # 对字符进行处理
      ```
      + 每次一行
      ```
      with open("somefile.txt") as f:
          while True:
              line = f.readline()
              if not line: 
                  break
              process(line) # 对行进行处理
      ```
      + 读取所有内容
      ```
      with open("somefile.txt") as f:
          for line in f.readlines(): 
              process(line) # 对行进行处理
      ```
      + 迭代文件
         1. 文件实际上可迭代, 直接在for循环中迭代行
         ```
         with open("somefile.txt") as f:
             for line in f:
                 process(line)
         ```
         2. sys.stdin也是可以迭代的
         ```
         import sys
         for line in sys.stdin:
             process(line)
         ```
      + print()写入文件
        
         1. print(写入内容, file=文件对象变量), 将文件内容写入一行, 并且自动添加换行符
      
# Chapter 16 测试基础
   + 先测试, 再编写
      + 测试驱动编程
         + 先编写测试, 再编写让测试通过的程序
   + 测试工具
      + doctest
         1. testmod()读取模块中的文档字符串, 查找类似交互式解释器中摘取的示例, 再检查这些示例是否反映了真实情况
         ```
         def square(x):
             '''
             :param x:
             :return: x的平方
             >>> square(2)
             4
             >>> square(3)
             9
             '''
             return x ** x

         if __name__ == "__main__":
             import test, doctest
             doctest.testmod(test)
         ```
         ```
         $ python test.py
         ```
         2. 代码通过测设, 则无信息报出, 如果没有通过测试, 返回错误信息
         ```
         Failed example:
             square(3)
         Expected:
             9
         Got:
             27
         ```
      + unittest
         ```
         import unittest, my_math

         class ProductTestCase(unittest.TestCase):

             def test_integers(self):
                 for x in range(-10, 10):
                     for y in range(-10, 10):
                         p = my_math.product(x, y)
                         self.assertEqual(p, x * y, 'Integer multiplication failed')

             def test_floats(self):
                 for x in range(-10, 10):
                     for y in range(-10, 10):
                         x = x / 10
                         y = y / 10
                         p = my_math.product(x, y)
                         self.assertEqual(p, x * y, 'Float multiplication failed')

         if __name__ == '__main__': unittest.main()
         ```
         1. unittest.main()会替你运行测试, 实例化TestCase子类, 并运行所有名称以test打头的方法
         2. 如果定义了setUp和tearDown, 它们将分别在每个测试方法之前和之后执行
         3. unittest还有很多其他测试方法, assertTrue, assertIsNotNone, assertAlmostEqual
         4. unittest区分错误和失败, 错误是引发异常, 失败的测试结果错误
   + 检查源代码
      + 一种发现代码中常见错误或问题的方式, 有点类似静态语言中的编译器
      + PyChecker, 能够找出给函数提供的参数不对等错误
      + PyLint, 支持PyChecker的大部分功能, 还可以检查变量名是否符合指定的命名规约, 是否符合编码标准
   + 性能分析
      + 标准库中包含性能分析模块profile, 以及一个c语言版本, cProfile
      + 只需要调用其方法run并提供一个字符串参数
      ```
      >>> def product(x, y):
      ···     return x*y
      >>> import cProfile
      >>> cProfile.run("product(2, 3)")
      4 function calls in 0.000 seconds
         Ordered by: standard name
         ncalls  tottime  percall  cumtime  percall filename:lineno(function)
              1    0.000    0.000    0.000    0.000 <input>:1(product)
              1    0.000    0.000    0.000    0.000 <string>:1(<module>)
              1    0.000    0.000    0.000    0.000 {built-in method builtins.exec}
              1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
      ```
      + run()还可以添加第二个参数, 一个文件名例如`testResult.profile`, 分析结果将会保存在这个文件中, 然后可以使用pstats模块分析结果
      ```
      >>> import pstats
      >>> p = pstats.Stats('testResult.profile')
      ```
      
# Chapter 17 扩展Python
# Chapter 19 配置 日志
   + 配置文件
      1. 将一些常量单独提取到一个python文件中, 使用python语法, `greeting = "Hello, world!"`, 这样结果中将会多两个引号
      2. configparser模块, 可以采用新的配置语法格式, 例如`greeting: Hello, world!`
      3. configparser模块的文件中必须使用`[files], [colors]`等标题将配置文件分成几部分
      ```
      # 配置文件 area.ini
      [numbers]
      PI: 3.14159265358
      [messages]
      greeting: welcome to the area calculation program!
      question: Please enter the radius:
      result_message: The area is 
      ```
      ```
      # 主程序
      from configparser import ConfigParser
      CONFIGFILE = "area.ini"
      config = ConfigParser()
      config.read(CONFIGFILE)
      
      # 打印默认问候语
      print(config['message'].get("greeting"))
      
      # 使用用户提示输入半径
      radius = float(input(config['message'].get("question")+' '))
      
      # 打印配置文件中的结果消息
      print(config[message].get('result_message') + ' ')
      
      # 打印结果
      print(config[numbers].get('PI')*radius**2 )
      ```
   + 日志
      + 使用logging模块
         1. 运行程序, 则会打印日志程序崩溃之前的信息到log文件中, 根据打印结果可以判断程序错误发生的地方
      ```
      import logging
      
      logging.basicConfig(level = logging.INFO, filename = "mylog.log")
      
      logging.info('starting program')
      logging.info('Trying to divide 1 by 0')
      
      print(1/0)
      
      logging.info('The division succeeded')
      logging.info('Ending program')
      ```
      + 日志等级
         + 默认六种日志等级, 括号中为级别对应数值
            1. NOTSET(0), DEBUG(10), INFO(20), WARNING(30), ERROR(40), CRITICAL(50)
            2. 自定义日志等级时, 不要与默认等级重复
            3. logging执行时输出大于等于设置日志等级的日志信息
      + 日志流程
         + logging包含的工作模块
            1. Logger:日志, 暴露函数给应用程序, 基于日志记录器和过滤器级别决定哪些日志有效
            2. LogRecord:日志记录器, 将日志传到相应的处理器处理
            3. Handler:处理器, 将(日志记录器产生的)日志记录发送至合适的目的地
            4. Filter:过滤器, 提供了更好的粒度控制,它可以决定输出哪些日志记录
            5. Formatter:格式化器, 指明了最终输出中日志记录的布局
         + 工作流程
            1. 判断Logger对象对于设置的级别是否可用, 如果可用, 则往下执行, 否则流程结束
            2. 创建LogRecord对象, 如果注册到Logger对象中的Filter对象过滤后返回False, 则不记录日志, 流程结束, 否则则向下执行
            3. LogRecord对象将Handler对象传入当前的Logger对象, 如果Handler对象的日志级别大于设置的日志级别, 再判断注册到Handler对象中的Filter对象过滤后是否返回True而放行输出日志信息, 否则不放行, 流程结束
            4. 如果传入的Handler大于Logger中设置的级别, 也即Handler有效, 则往下执行, 否则流程结束
            5. 判断这个Logger对象是否还有父Logger对象, 如果没有, 代表当前Logger对象是最顶层的Logger对象, root Logger, 流程结束; 否则将Logger对象设置为它的父Logger对象, 重复上面的3、4两步，输出父类Logger对象中的日志输出，直到是root Logger为止。
      + 日志输出格式
        
         1. `日志级别 : Logger实例名称 : 日志消息内容`
      + 基本使用
         1. 使用logging.basicConfig()来设置日志
         2. ***默认的日志级别被设置为WARNING***
            
            | 参数名称 | 参数描述 |
            |---------|--------|
            | filename | 日志输出到文件的文件名 |
            | filemode | 日志文件打开模式, r[+], w[+], a[+] |
            | format | 日志输出格式, `%(输出内容)s` |
            | datefmt | 日志附带日期时间的格式 |
            | style | 格式占位符，默认为 "%" 和 "{}" |
            | level | 日志输出级别 |
            | stream | 定义输出流, 用来初始化StreamHandler对象, 不能filename参数一起使用, 否则会ValueError异常 |
            | handles | 定义处理器, 用来创建Handler对象, 不能和filename, stream参数一起使用, 否则也会抛出ValueError异常 |
            
            ```
            import logging
            
            logging.basicConfig()
            logging.debug("This is a debug message")
            logging.info("This is an info message")
            logging.warning("This is a warning message")
            logging.error("This is an error message")
            logging.critical("This is a critical message")
            ```
            ```
            # 输出结果如下
            WARNING:root:This is a warning message
            ERROR:root:This is an error message
            CRITICAL:root:This is a critical message
            ```
         3. 对logging的基本设置
            ```
            import logging
            
            logging.basicConfig(filename='mylog.log', filemode='w', level=logging.DEBUG, format= "%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")
            logging.debug("This is a debug message")
            logging.info("This is an info message")
            logging.warning("This is a warning message")
            logging.error("This is an error message")
            logging.critical("This is a critical message")
            ```
            ```
            # 输出结果如下
            24-55-2020 20:55:51 root:DEBUG:This is a debug message
            24-55-2020 20:55:51 root:INFO:This is an info message
            24-55-2020 20:55:51 root:WARNING:This is a warning message
            24-55-2020 20:55:51 root:ERROR:This is an error message
            24-55-2020 20:55:51 root:CRITICAL:This is a critical message
            ```
         4. 出现异常时, 日志的debug(), warning(), error()等函数需要设置参数
         ```
      import logging
      
         logging.basicConfig(filename="test.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
         a = 5
         b = 0
         try:
             c = a / b
         except Exception as e:
             # 以下三条语句的效果相同
             logging.exception("Exception occurred")
             logging.error("Exception occurred", exc_info=True)
             logging.log(level=logging.DEBUG, msg="Exception occurred", exc_info=True)
         ```
      + 自定义Logger
         1. 整个系统只有一个根Logger对象, Logger对象在执行info(), error()等方法时实际上调用都是根Logger对象对应的info(), error()等方法, 我们可以创造多个Logger对象, 但是真正输出日志的是根Logger对象, 每个Logger对象都可以设置一个名字
         2. 获取Logger对象的方法是getLogger()
         ```
         logger = logging.getLogger(__name__)
         ```
   3. format常用变量格式
      
        | 变量 | 格式 | 变量描述 |
        |-----|------|--------|
        | asctime | %(asctime)s | 将日志的时间构造成可读的形式, 默认情况下是精确到毫秒, 如2018-10-13 23:24:57,832, 可以额外指定datefmt参数来指定该变量的格式 |
        | name | %(name)s | Logger的名称 |
        | filename | %(filename)s | 不包含路径的程序文件名称 |
        | pathname | %(pathname)s | 包含路径的程序文件名称 |
        | funcName | %(funcName)s | 日志记录所在的函数名 |
        | levelname | %(levelname)s | 日志的级别名称 |
        | message | %(message)s | 具体的日志信息 |
        | lineno | %(lineno)d | 日志记录所在的行号 |
        | process | %(process)d | 当前进程ID |
        | processName | %(processName)s | 当前进程名称 |
        | thread | %(thread)d | 当前线程ID |
        | threadName | %(threadName)s | 当前线程名称 |
        
        4. Logger对象和Handler对象都可以设置级别, 而默认Logger对象级别为WARNING, 默认Handler对象级别为NOTSET
        5. 既想在控制台中输出DEBUG级别的日志, 又想在文件中输出WARNING级别的日志, 可以只设置一个最低级别的Logger对象, 两个不同级别的Handler对象  
        ```
        import logging
        import logging.handlers

        logger = logging.getLogger("logger")
            
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(filename="mylog.log")
            
        logger.setLevel(logging.DEBUG)
        handler1.setLevel(logging.WARNING)
        handler2.setLevel(logging.DEBUG)
            
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
            
        logger.addHandler(handler1)
        logger.addHandler(handler2)
            
        # 分别为 10、30、30
        # print(handler1.level)
        # print(handler2.level)
        # print(logger.level)

        logger.debug('This is a customer debug message')
        logger.info('This is an customer info message')
        logger.warning('This is a customer warning message')
        logger.error('This is an customer error message')
        logger.critical('This is a customer critical message')
        ```
