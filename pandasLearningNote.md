# pandas学习笔记
## Chapter 1 pandas入门
   + pandas核心就像操作一个电子表格的无头版本，比如Excel。使用的大多数数据集将是所谓的数据帧（DataFrame），数据帧通常就像电子表格一样，拥有列和行
   ```
   import datetime
   import pandas_datareader.data as web
   import matplotlib.pyplot as plt
   from matplotlib import style

   style.use('fivethirtyeight')

   start = datetime.datetime(2010, 1, 1)
   end = datetime.datetime(2015, 12, 31)

   df = web.DataReader("XOM", "morningstar", start, end)

   df.reset_index(inplace=True)
   df.set_index("Date", inplace=True)
   df = df.drop("Symbol", axis=1)

   print(df.head())

   df['High'].plot()
   plt.legend()
   plt.show()
   ```

## Chapter 2 pandas基础
   1. 加载到Pandas数据帧之前，数据可能有多种形式，但通常需要是以行和列组成的数据集。 所以也许是这样的字典：
       ```
       web_stats = {'Day':[1,2,3,4,5,6],
                    'Visitors':[43,34,65,56,29,76],
                    'Bounce Rate':[65,67,78,65,45,52]}
       ```
      + 可以将这个字典转换成数据帧, 并且查看前几行或者后几行
       ```
       import pandas as pd
       
       web_stats = {'Day':[1,2,3,4,5,6],
                    'Visitors':[43,34,65,56,29,76],
                    'Bounce Rate':[65,67,78,65,45,52]}   
       df = pd.DataFrame(web_stats)
       print(df.head())
       #    Bounce Rate  Day  Visitors
       # 0           65    1        43
       # 1           67    2        34
       # 2           78    3        65
       # 3           65    4        56
       # 4           45    5        29
       print(df.tail())
       #    Bounce Rate  Day  Visitors
       # 1           67    2        34
       # 2           78    3        65
       # 3           65    4        56
       # 4           45    5        29
       # 5           52    6        76
       ```
      + 也可以查看头部或者尾部的几行, 只需在`head()`或者`tail()`中传入参数
   2. 索引
      + 左边有这些数字，`0,1,2,3,4,5`等等，就像行号一样。这些数字实际上是你的“索引”。数据帧的索引是数据相关，或者数据按它排序的东西。一般来说，这将是连接所有数据的变量。
      + 当你没有定义索引时，Pandas 会像这样为你生成一个。也可以自定义索引。
         1. 在任何现有的数据帧上，我们可以像这样设置一个新的索引
         ```
         df.set_index('Day', inplace=True)
              # Bounce Rate  Visitors
         # Day                       
         # 1             65        43
         # 2             67        34
         # 3             78        65
         # 4             65        56
         # 5             45        29
         ```
         2. `inplace = True`将会修改原来的数据帧，即修改了变量本身；如果没有，则会返回一个新的数据帧是修改的结果
         ```
         >>> df.set_index('Day')
              Visitors  Bounce Rate
         Day                       
         1          43           65
         2          34           67
         3          65           78
         4          56           65
         5          29           45
         6          76           52
         ```
         3. 引用特定的列，使用列名称 `df['Visitors']`
         ```
         >>> print(df['Visitors'])
         Day
         1    43
         2    34
         3    65
         4    56
         5    29
         6    76
         Name: Visitors, dtype: int64
         ```
         4. 引用数据帧多列
         ```
         print(df[['Visitors','Bounce Rate']])
         ```
   3. 绘图
      + 绘制单列
      ```
      >>> df['Visitors'].plot()
      >>> plt.show()
      ```
      + 绘制整个数据帧
      ```
      >>> df.plot()
      >>> plt.show()
      ```
## Chapter 3 IO基础
   1. 读入数据
   ```
   import pandas as pd
   
   df = pd.read_excel("test.xlsx")
   print(df.head())
   ```
   2. 导出数据
   ```
   df.set_index('Level1', inplace=True)
   # 导出所有数据
   df.to_excel("new.xlsx")
   # 导出某一列数据
   df["Question"].to_excel("Only_Question.xlsx")
   # 导出多列数据
   df[["Question", "Level2"]].to_excel("2Columns.xlsx")
   ```
   3. 导入时设置索引, ***index_col***
   ```
   df = pd.read_csv('newcsv2.csv', index_col=0)
   print(df.head())
   ```
   4. 读入导出数据时表头设置 `df.columns = ["表头新名称"]`
   ```
   df = pd.read_excel("2Columns.xlsx")
   print(df.head())
   df.columns = ["Col1", "Col2", "Col3"]
   ```
   5. 保存文件表头设置  ***header=False***
   ```
   df.to_csv('newcsv4.csv', header=False)
   ```
   6. 导入文件的表头设置
   ```
   df = pd.read_csv('newcsv4.csv', names = ['Date','House_Price'], index_col=0)
   print(df.head())
   ```
   7. 表头的重命名 `df.rename(columns={})`
   ```
   print(df.head())
   df.rename(columns={'House_Price':'Prices'}, inplace=True)
   print(df.head())
   #      Date  House_Price
   # 0  2015-06-30       502300
   # 1  2015-05-31       501500
   # 2  2015-04-30       500100
   # 3  2015-03-31       495800
   # 4  2015-02-28       492700
   #      Date  Prices
   # 0  2015-06-30  502300
   # 1  2015-05-31  501500
   # 2  2015-04-30  500100
   # 3  2015-03-31  495800
   # 4  2015-02-28  492700
   ```
## Chapter 4 构建数据帧
   1. 与html文件转换 `pd.read_html('网址') df.to_html('文件名')`
   2. dataframe类似于列表, 索引从0开始, 提取某一行或者列, `df.loc[行索引, 列表头名称], df.iloc[行索引, 列索引]`
   ```
   print(df)
   print(df['Level2'])
   print(df.loc[:, 'Level2'])
   print(df.iloc[:, 0])
   ```
## Chapter 5 组合数据帧
   1. `concat([数据帧列表])` 简单的索引的拼接, 当出现不同列时, 使用NaN填充没有数据的地方
   ```
   df1 = pd.DataFrame({'HPI':[80,85,88,85],
                       'Int_rate':[2, 3, 2, 2],
                       'US_GDP_Thousands':[50, 55, 65, 55]},
                      index = [2001, 2002, 2003, 2004])

   df2 = pd.DataFrame({'HPI':[80,85,88,85],
                       'Int_rate':[2, 3, 2, 2],
                       'US_GDP_Thousands':[50, 55, 65, 55]},
                      index = [2005, 2006, 2007, 2008])

   df3 = pd.DataFrame({'HPI':[80,85,88,85],
                       'Int_rate':[2, 3, 2, 2],
                       'Low_tier_HPI':[50, 52, 50, 53]},
                      index = [2001, 2002, 2003, 2004])
   # 有相同列, 索引按照列表的数据帧顺序简单拼接                  
   pd.concat([df1, df2]) 
   #       HPI  Int_rate  US_GDP_Thousands
   # 2001   80         2                50
   # 2002   85         3                55
   # 2003   88         2                65
   # 2004   85         2                55
   # 2005   80         2                50
   # 2006   85         3                55
   # 2007   88         2                65
   # 2008   85         2                55
   
   # 出现不同列, NaN填充数据
   pd.concat([df1, df2, df3])
   #       HPI  Int_rate  Low_tier_HPI  US_GDP_Thousands
   # 2001   80         2           NaN              50.0
   # 2002   85         3           NaN              55.0
   # 2003   88         2           NaN              65.0
   # 2004   85         2           NaN              55.0
   # 2005   80         2           NaN              50.0
   # 2006   85         3           NaN              55.0
   # 2007   88         2           NaN              65.0
   # 2008   85         2           NaN              55.0
   # 2001   80         2          50.0               NaN
   # 2002   85         3          52.0               NaN
   # 2003   88         2          50.0               NaN
   # 2004   85         2          53.0               NaN
   ```
   2. `df.append()` 附加到数据帧之后返回一个新的数据帧, 如果索引相同也会附加，不会合并相同索引, 列不相同用NaN填充
   ```
   df1.append(df2)
   #       HPI  Int_rate  US_GDP_Thousands
   # 2001   80         2                50
   # 2002   85         3                55
   # 2003   88         2                65
   # 2004   85         2                55
   # 2005   80         2                50
   # 2006   85         3                55
   # 2007   88         2                65
   # 2008   85         2                55
   df1.append(df3)
   #       HPI  Int_rate  Low_tier_HPI  US_GDP_Thousands
   # 2001   80         2           NaN              50.0
   # 2002   85         3           NaN              55.0
   # 2003   88         2           NaN              65.0
   # 2004   85         2           NaN              55.0
   # 2001   80         2          50.0               NaN
   # 2002   85         3          52.0               NaN
   # 2003   88         2          50.0               NaN
   # 2004   85         2          53.0               NaN
   ```
   3. 将序列添加到数据帧
   ```
   s = pd.Series([80, 50, 2], index=['HPI','Int_rate','US_GDP_Thousands'])
   df1.append(s, ignore_index=True)
   #    HPI  Int_rate  US_GDP_Thousands
   # 0   80         2                50
   # 1   85         3                55
   # 2   88         2                65
   # 3   85         2                55
   # 4   80         2                50
   ```
   
## Chapter 6 合并数据帧
   1. 合并数据帧 pd.merge()
   ```
   df1 = pd.DataFrame({'HPI':[80,85,88,85],
                       'Int_rate':[2, 3, 2, 2],
                       'US_GDP_Thousands':[50, 55, 65, 55]},
                      index = [2001, 2002, 2003, 2004])
   df2 = pd.DataFrame({'HPI':[80,85,88,85],
                       'Int_rate':[2, 3, 2, 2],
                       'US_GDP_Thousands':[50, 55, 65, 55]},
                      index = [2005, 2006, 2007, 2008])
   df3 = pd.DataFrame({'HPI':[80,85,88,85],
                       'Unemployment':[7, 8, 9, 6],
                       'Low_tier_HPI':[50, 52, 50, 53]},
                      index = [2001, 2002, 2003, 2004])
                      
   # 在HPI这一共有列合并  结果排列了所有可能的组合情况             
   print(pd.merge(df1,df3, on='HPI'))
   #    HPI  Int_rate  US_GDP_Thousands  Low_tier_HPI  Unemployment
   # 0   80         2                50            50             7
   # 1   85         3                55            52             8
   # 2   85         3                55            53             6
   # 3   85         2                55            52             8
   # 4   85         2                55            53             6
   # 5   88         2                65            50             9
   
   # 共有两个列上合并 
   print(pd.merge(df1,df2, on=['HPI','Int_rate']))
   #    HPI  Int_rate  US_GDP_Thousands_x  US_GDP_Thousands_y
   # 0   80         2                  50                  50
   # 1   85         3                  55                  55
   # 2   88         2                  65                  65
   # 3   85         2                  55                  55
   # 注意这里有US_GDP_Thousands的两个版本。这是因为我们没有共享这些列，所以都保留下来，使用另外一个字母来区分。
   ```
   2. join() 合并是根据索引进行合并， 所以需要指定索引列
   ```
   df1.set_index('HPI', inplace=True)
   df3.set_index('HPI', inplace=True)

   joined = df1.join(df3)
   #      Int_rate  US_GDP_Thousands  Low_tier_HPI  Unemployment
   # HPI                                                        
   # 80          2                50            50             7
   # 85          3                55            52             8
   # 85          3                55            53             6
   # 85          2                55            52             8
   # 85          2                55            53             6
   # 88          2                65            50             9
   ```
   3. merge()和join()会合并共有列的共同数据或者左连接或者右连接, 会有不同合并方式`left, right, inner, outer`, `inner`是使用共有列的数据交集, `outer`是使用并集
   ```
   df1 = pd.DataFrame({
                       'Int_rate':[2, 3, 2, 2],
                       'US_GDP_Thousands':[50, 55, 65, 55],
                       'Year':[2001, 2002, 2003, 2004]
                       })

   df3 = pd.DataFrame({
                       'Unemployment':[7, 8, 9, 6],
                       'Low_tier_HPI':[50, 52, 50, 53],
                       'Year':[2001, 2003, 2004, 2005]})
   
   # 只合并了共有列的相同数据, 不同数据被舍弃                    
   pd.merge(df1,df3, on='Year')
   #    Int_rate  US_GDP_Thousands  Year  Low_tier_HPI  Unemployment
   # 0         2                50  2001            50             7
   # 1         2                65  2003            52             8
   # 2         2                55  2004            50             9
   
   # 按照df1的共有列合并， NaN填充数据
   pd.merge(df1, df3, on="Year", how="left")
   #    Int_rate  US_GDP_Thousands  Year  Unemployment  Low_tier_HPI
   # 0         2                50  2001           7.0          50.0
   # 1         3                55  2002           NaN           NaN
   # 2         2                65  2003           8.0          52.0
   # 3         2                55  2004           9.0          50.0
   
   # 按照df3的共有列合并， NaN填充数据
   pd.merge(df1, df3, on="Year", how="right")
   #    Int_rate  US_GDP_Thousands  Year  Unemployment  Low_tier_HPI
   # 0       2.0              50.0  2001             7            50
   # 1       2.0              65.0  2003             8            52
   # 2       2.0              55.0  2004             9            50
   # 3       NaN               NaN  2005             6            53
   ```
   
## Chapter 7 Pickle
   1. 用于将二进制数据保存到可以稍后访问的文件。在Python中，这被称为Pickle。Python有一个名为Pickle的模块，它将把一个对象转换成一个字节流，或者反过来转换它。这可以保存任何Python对象。那机器学习分类器，字典，数据帧
   ```
   import pickle
   pickle_out = open("test.pickle", 'wb')
   pickle.dump(df1, pickle_out)
   pickle_out.close()
   ```
   2. 读取数据
   ```
   pickle_in = open("test.pickle", "rb")
   df = pickle.load(pickle_in)
   pickle_in.close()
   ```
   3. pandas模块自带有pickle
   ```
   HPI_data.to_pickle('pickle.pickle')
   HPI_data2 = pd.read_pickle('pickle.pickle')
   print(HPI_data2)
   ```

## Chapter 8 百分比变化和相关表
   1. 某一列的改变
   ```
   #       Int_rate  US_GDP_Thousands
   # Year                            
   # 2001         2                50
   # 2002         3                55
   # 2003         2                65
   # 2004         2                55
   
   df['Int_rate2'] = df['Int_rate']*2
   #       Int_rate  US_GDP_Thousands  Int_rate2
   # Year                                       
   # 2001         2                50          4
   # 2002         3                55          6
   # 2003         2                65          4
   # 2004         2                55          4   
   ```
   2. 根据索引, 百分比的变化量
   ```
   #       Int_rate  US_GDP_Thousands  Int_rate2
   # Year                                       
   # 2001         2                50          4
   # 2002         3                55          6
   # 2003         2                65          4
   # 2004         2                55          4  
   
   df.pct_change()
   #       Int_rate  US_GDP_Thousands  Int_rate2
   # Year                                       
   # 2001       NaN               NaN        NaN
   # 2002  0.500000          0.100000   0.500000
   # 2003 -0.333333          0.181818  -0.333333
   # 2004  0.000000         -0.153846   0.000000
   ```
   3. 计算协方差 `df.corr()`
   ```
   df.corr()
   #                  Int_rate  US_GDP_Thousands  Int_rate2
   # Int_rate          1.000000         -0.132453   1.000000
   # US_GDP_Thousands -0.132453          1.000000  -0.132453
   # Int_rate2         1.000000         -0.132453   1.000000
   ```
   4. 每一列数据的各项指标 `df.describe()`
   ```
   df.describe()
   #        Int_rate  US_GDP_Thousands  Int_rate2
   # count      4.00          4.000000        4.0
   # mean       2.25         56.250000        4.5
   # std        0.50          6.291529        1.0
   # min        2.00         50.000000        4.0
   # 25%        2.00         53.750000        4.0
   # 50%        2.00         55.000000        4.0
   # 75%        2.25         57.500000        4.5
   # max        3.00         65.000000        6.0
   ```

## Chapter 9 重采样
   1. df.resample() 每隔一段时间重新采样
      + 重采样时间间隔的选项说明 [时间间隔选项](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects)
   ```
   TX1yr = HPI_data['TX'].resample('A')
   print(TX1yr.head())
   ```
   2. 重采样的方式, 默认是时间间隔内的平均值, 也可以是`sum, mean, std, sem, max, min, median, first, last, ohlc`, ohlc代表起始值，最高值，最低值和最后一个值
   ```
   df.resample('5Min').mean()
   ```

## Chapter 10 处理缺失数据
   1. 删除包含NaN的数据 `df.dropna()`
   ```
   #       HPI  Int_rate  US_GDP_Thousands  Unemployment  Low_tier_HPI
   # 2005   80         2                50           6.0          53.0
   # 2006   85         3                55           NaN           NaN
   # 2007   88         2                65           NaN           NaN
   # 2008   85         2                55           NaN           NaN
   df4.dropna()
   #       HPI  Int_rate  US_GDP_Thousands  Unemployment  Low_tier_HPI
   # 2005   80         2                50           6.0          53.0
   ```
   2. 只在整行NaN时删除 `df.dropna(how='all')`
   3. 根据非NaN的个数进行删除, `df.dropna(how='any', thresh=x)`, x为非NaN的个数
   4. 前向填充, 即将数据向前扫描，填充到缺失的数据中`fillna(method='ffill')`, 把它看作是一个扫描动作，其中你可以从过去获取数据，将其转移到缺失的数据中
   ```
   #       HPI  Int_rate  US_GDP_Thousands  Unemployment  Low_tier_HPI
   # 2005   80         2                50           6.0          53.0
   # 2006   85         3                55           NaN           NaN
   # 2007   88         2                65           NaN           NaN
   # 2008   85         2                55           NaN           NaN   
   
   df4.fillna(method='ffill')
   #       HPI  Int_rate  US_GDP_Thousands  Unemployment  Low_tier_HPI
   # 2005   80         2                50           6.0          53.0
   # 2006   85         3                55           6.0          53.0
   # 2007   88         2                65           6.0          53.0
   # 2008   85         2                55           6.0          53.0   
   ```
   5. Bfill或后向填充是相反的 `fillna(method='bfill')`
   6. 替换NaN值, `fillna(value=-9999)`, 使用value替换
   ```
   df4.fillna(value=-9999)
   #       HPI  Int_rate  US_GDP_Thousands  Unemployment  Low_tier_HPI
   # 2005   80         2                50           6.0          53.0
   # 2006   85         3                55       -9999.0       -9999.0
   # 2007   88         2                65       -9999.0       -9999.0
   # 2008   85         2                55       -9999.0       -9999.0   
   ```
   
## Chapter 11 滚动统计量
   1. 移动均值 移动均值就是当前值加上前面一段时间的数据的均值 `rolling(执行周期).mean()`, 使用滚动统计量，开头将生成NaN数据
   ```
   #         Int_rate  US_GDP_Thousands
   # Year                            
   # 2001         2                50
   # 2002         3                55
   # 2003         2                65
   # 2004         2                55
   df1["Int_rate"].rolling(2).mean()
   # Year
   # 2001    NaN
   # 2002    2.5
   # 2003    2.5
   # 2004    2.0
   # Name: Int_rate, dtype: float64
   ```
   2. `rolling().std()`移动标准差
   3. `rolling().corr()`移动协方差
   4. `rolling().apply()`, 可以应用自己写的移动窗函数
   ```
   def moving_average(values):
       ma = mean(values)
       return ma
       
   df.rolling(2).apply(moving_average)
   ```
   
## Chapter 12 比较操作
   1. 类似于lamda表达式的方式, 在DataFrame加入条件
   ```
   bridge_height = {'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
   df = pd.DataFrame(bridge_height)
   
   #      meters
   # 0    10.26
   # 1    10.31
   # 2    10.27
   # 3    10.22
   # 4    10.23
   # 5  6212.42
   # 6    10.28
   # 7    10.25
   # 8    10.31
   
   df["STD"] = df.rolling(2).std()
   #     meters          STD
   # 0    10.26          NaN
   # 1    10.31     0.035355
   # 2    10.27     0.028284
   # 3    10.22     0.035355
   # 4    10.23     0.007071
   # 5  6212.42  4385.610607
   # 6    10.28  4385.575252
   # 7    10.25     0.021213
   # 8    10.31     0.042426
   
   df_std = df.describe()['meters']['std'] # 2067.3845835687607
   
   df = df[ (df['STD'] < df_std) ]
   #    meters       STD
   # 1   10.31  0.035355
   # 2   10.27  0.028284
   # 3   10.22  0.035355
   # 4   10.23  0.007071
   # 7   10.25  0.021213
   # 8   10.31  0.042426
   ```
   
## Chapter 13 Scikit Learn 交互
   1. Scikit-learn(sklearn)是机器学习中常用的第三方模块，对常用的机器学习方法进行了封装，包括回归(Regression)、降维(Dimensionality Reduction)、分类(Classfication)、聚类(Clustering)等方法。
   2. `drop(label, axis)`, 删除行或者列, label为行列的标签, axis为0时表示删除行，axis为1时表示删除列
   ```
   from sklearn import svm, preprocessing, cross_validation
   
   X = np.array(housing_data.drop(['label','US_HPI_future'], 1))
   X = preprocessing.scale(X)
   y = np.array(housing_data['label'])
   
   X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
   clf = svm.SVC(kernel='linear')
   clf.fit(X_train, y_train)
   print(clf.score(X_test, y_test))
   ```
   
   
   
   
       
   

   
   
   
   
   