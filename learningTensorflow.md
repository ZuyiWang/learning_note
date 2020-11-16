# TensorFlow学习笔记
## Chapter1 基本使用
   1. 基本概念
      + 使用图 graph 来表示计算任务
      + 在会话 session 的上下文中执行图
      + 使用张量 tensor 表示数据
      + 使用变量 Variable 表示维护状态
      + 使用feed和fetch可以为任意操作赋值或者从中获取数据
   2. 综述
      + TensorFlow是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为op(operation). 一个op获得0个或多个Tensor, 执行计算, 产生0个或多个Tensor. 每个Tensor是一个类型化的多维数组. 例如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是`[batch, height, width, channels]`.
      + 一个TensorFlow图描述了计算的过程. 为了进行计算, 图必须在会话里被启动. 会话将图的op分发到诸如CPU或GPU之类的设备上, 同时提供执行op的方法. 这些方法执行后, 将产生的tensor返回. 在Python语言中, 返回的tensor是numpy ndarray 对象; 在C和C++语言中, 返回的tensor是tensorflow::Tensor实例.
   3. 计算图
      + TensorFlow程序通常被组织成一个构建阶段和一个执行阶段. 在构建阶段, op的执行步骤 被描述成一个图. 
      + 在执行阶段, 使用会话执行执行图中的op. 例如, 通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练op.
   4. 构建图   
      + 构建图的第一步, 是创建源op(source op). 源op不需要任何输入, 例如常量(Constant). 源op的输出被传递给其它op做运算.
      + Python 库中, op构造器的返回值代表被构造出的op的输出, 这些返回值可以传递给其它op构造器作为输入.
      + TensorFlow Python 库有一个默认图(default graph), op构造器可以为其增加节点. 
      ```
      import tensorflow.copmpat.v1 as tf
      tf.diable_v2_behavior()
      
      # 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
      # 加到默认图中.
      #
      # 构造器的返回值代表该常量 op 的返回值.
      matrix1 = tf.constant([[3., 3.]])

      # 创建另外一个常量 op, 产生一个 2x1 矩阵.
      matrix2 = tf.constant([[2.],[2.]])

      # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
      # 返回值 'product' 代表矩阵乘法的结果.
      product = tf.matmul(matrix1, matrix2)
      ```
   5. 启动会话
      + 构造阶段完成后, 才能启动图. 启动图的第一步是创建一个 Session 对象, 如果无任何创建参数, 会话构造器将启动默认图.
      ```
      # 启动默认图.
      sess = tf.Session()

      # 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数. 
      # 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
      # 矩阵乘法 op 的输出.
      #
      # 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
      # 
      # 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
      #
      # 返回值 'result' 是一个 numpy `ndarray` 对象.
      result = sess.run(product)
      print(result)
      # ==> [[ 12.]]

      # 任务完成, 关闭会话.
      sess.close()
      ```
      + Session对象在使用完后需要关闭以释放资源. 除了显式调用close外, 也可以使用 "with" 代码块 来自动完成关闭动作.
      ```
      with tf.Session() as sess:
          result = sess.run([product])
          print(result)
      ```
   6. 指定运算设备CPU或GPU
      + 一般不需要显式指定使用CPU还是GPU, TensorFlow能自动检测. 如果检测到GPU, TensorFlow会尽可能地利用找到的第一个GPU来执行操作.
      + 如果机器上有超过一个可用的GPU, 除第一个外的其它GPU默认是不参与计算的. 为了让TensorFlow使用这些GPU, 你必须将op明确指派给它们执行. with...Device 语句用来指派特定的CPU或GPU执行操作:
      ```
      with tf.Session() as sess:
          with tf.device("/gpu:1"):
              matrix1 = tf.constant([[3., 3.]])
              matrix2 = tf.constant([[2.],[2.]])
              product = tf.matmul(matrix1, matrix2)
      ```
      + 设备用字符串进行标识
         + `"/cpu:0"`: 机器的 CPU.
         + `"/gpu:0"`: 机器的第一个 GPU, 如果有的话.
         + `"/gpu:1"`: 机器的第二个 GPU, 以此类推.
   7. 交互式界面使用
      + 为了便于使用诸如IPython之类的Python交互环境, 可以使用InteractiveSession代替Session类, 使用Tensor.eval()和Operation.run()方法代替Session.run(). 这样可以避免使用一个变量来持有会话.
      ```
      # 进入一个交互式 TensorFlow 会话.
      import tensorflow.compat.v1 as tf
      tf.disable_v2_behavior()
      sess = tf.InteractiveSession()

      x = tf.Variable([1.0, 2.0])
      a = tf.constant([3.0, 3.0])

      # 使用初始化器 initializer op 的 run() 方法初始化 'x' 
      x.initializer.run()

      # 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
      sub = tf.sub(x, a)
      print(sub.eval())
      # ==> [-2. -1.]
      ```
   8. Tensor 张量
      + TensorFlow程序使用tensor数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是tensor. 你可以把TensorFlow tensor看作是一个n维的数组或列表. 一个tensor包含一个静态类型rank 阶数, 和一个shape 形状.
      + 变量 Variables，变量维护图执行过程中的状态信息. 
      + 通常会将一个统计模型中的参数表示为一组变量. 例如, 你可以将一个神经网络的权重作为某个变量存储在一个tensor中. 在训练过程中, 通过重复运行训练图, 更新这个tensor.
      ```
      # 创建一个变量, 初始化为标量 0.
      state = tf.Variable(0, name="counter")

      # 创建一个 op, 其作用是使 state 增加 1

      one = tf.constant(1)
      new_value = tf.add(state, one)
      update = tf.assign(state, new_value)

      # 启动图后, 变量必须先经过`初始化` (init) op 初始化,
      # 首先必须增加一个`初始化` op 到图中.
      init_op = tf.initialize_all_variables()

      # 启动图, 运行 op
      with tf.Session() as sess:
        # 运行 'init' op
        sess.run(init_op)
        # 打印 'state' 的初始值
        print sess.run(state)
        # 运行 op, 更新 'state', 并打印 'state'
        for _ in range(3):
          sess.run(update)
          print sess.run(state)

      # 输出:

      # 0
      # 1
      # 2
      # 3
      ```
   9. Fetch结果
      + 为了取回操作的输出内容, 可以在使用Session对象的run()调用执行图时, 传入一些tensor, 这些tensor会帮助你取回结果. 在之前的例子里, 我们只取回了单个节点state, 但是你也可以取回多个tensor:
      ```
      input1 = tf.constant(3.0)
      input2 = tf.constant(2.0)
      input3 = tf.constant(5.0)
      intermed = tf.add(input2, input3)
      mul = tf.mul(input1, intermed)
      
      with tf.Session() as sess:
          result = sess.run([mul, intermed])
          print(result)
      # 输出:
      # [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
      ```
   10. Feed变量
      + TensorFlow还提供了feed机制, 该机制可以临时替代图中的任意操作中的tensor可以对图中任何操作提交补丁, 直接插入一个tensor.
      + feed使用一个tensor值临时替换一个操作然后输出结果. 你可以提供feed数据作为run()调用的参数. feed只在调用它的方法内有效, 方法结束, feed就会消失. 最常见的用例是将某些特殊的操作指定为"feed"操作, 标记的方法是使用tf.placeholder()为这些操作创建占位符.
      + 如果没有正确提供feed, `placeholder()`操作将会产生错误. 
      ```
      input1 = tf.placeholder(tf.types.float32)
      input2 = tf.placeholder(tf.types.float32)
      output = tf.mul(input1, input2)

      with tf.Session() as sess:
        print sess.run([output], feed_dict={input1:[7.], input2:[2.]})

      # 输出:
      # [array([ 14.], dtype=float32)]
      ```
      
      
## Chapter2 Mnist机器学习入门
   1. Mnist数据集合下载
      1. 下载`input_data.py`文件，代码中`maybe_download()` 函数可以确保这些训练数据下载到本地文件夹中。文件夹的名字在 fully_connected_feed.py 文件的顶部由一个标记变量指定，你可以根据自己的需要进行修改。
      2. 
      |数据集合|目的|
      |--|--|
      |data_sets.train|55000组图片和标签，用于训练|
      |data_sets.validation|5000组图片和标签, 用于迭代验证训练的准确性。|
      |data_sets.test|10000组图片和标签，用于最终测试训练的准确性|
      3. 执行`read_data_sets()`函数将会返回一个DataSet实例，其中包含了以上三个数据集。函数`DataSet.next_batch()`是用于获取以batch_size为大小的一个元组，其中包含了一组图片和标签，该元组会被用于当前的TensorFlow运算会话中。
      ```
      images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
      ```
   2. 导入tensorflow
      1. `import tensorflow as tf`
      2. 使用2.0中的v1兼容包来沿用1.x代码
         + TensorFlow 2.0中提供了tensorflow.compat.v1代码包来兼容原有1.x的代码，可以做到几乎不加修改的运行。社区的contrib库因为涉及大量直接的TensorFlow引用代码或者自己写的Python扩展包，所以无法使用这种模式。TensorFlow 2.0中也已经移除了contrib库。使用这种方式升级原有代码，只需要把原有程序开始的TensorFlow引用改为:
      ```
      import tensorflow.compat.v1 as tf
      tf.disable_v2_behavior()
      ```
         + 使用迁移工具自动迁移1.x代码到2.0
      ```
      tf_upgrade_v2 --infile first-tf.py --outfile first-tf-v2.py
      ```

   3. 单层Softmax实现Mnist数字分类
      1. 为了得到一张给定图片属于某个特定数字类的证据（evidence），我们对图片像素值进行加权求和。如果这个像素具有很强的证据说明这张图片不属于该类，那么相应的权值为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值是正数。我们也需要加入一个额外的偏置量（bias），因为输入往往会带有一些无关的干扰量。因此对于给定的输入图片 x 它代表的是数字 i 的证据可以表示为
      ```
      evidence = WX+b
      ```
      2. 利用softmax函数进行正则化，输出属于各个数字的概率
      ```
      y = softmax(evidence)
      ```
      3. 设置输入参数 tf.placeholder()
         + `tf.compat.v1.placeholder(dtype, shape=None, name=None)`，shape是一个序列，None表示为任意维度。
         + 在运行sess.run()时，需要将placeholder的参数feed，以字典形式`feed_dict={name: value}`
      4. 设置可调参数 tf.Variable()
        + `tf.Variable(initial_value=None, trainable=None, validate_shape=True, caching_device=None,name=None, variable_def=None, dtype=None,import_scope=None,constraint=None,synchronization=tf.VariableSynchronization.AUTO,aggregation=tf.compat.v1.VariableAggregation.NONE, shape=None)`, [官方文档 v2.0](https://www.tensorflow.org/api_docs/python/tf/Variable)
        + 代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。
      5. 张量乘法 tf.matmul()
      6. 张量元素的加法 tf.reduce_sum()
         + `tf.compat.v1.reduce_sum(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None,keep_dims=None)`
         + 沿着axis所指定的方向，对输入张量进行求和
         ```
         x = tf.constant([[1, 1, 1], [1, 1, 1]])
         tf.reduce_sum(x)  # 6
         tf.reduce_sum(x, 0)  # [2, 2, 2]
         tf.reduce_sum(x, 1)  # [3, 3]
         tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
         tf.reduce_sum(x, [0, 1])  # 6
         ```
      7. softmax层的函数 tf.nn.softmax()
         + `tf.nn.softmax(logits, axis=None, name=None)`, 相当于执行操作`softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)`
         + 参数: 
            + logits: A non-empty Tensor. Must be one of the following types: half, float32, float64.
            + axis: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
            + name: A name for the operation (optional).
      8. 梯度下降法训练 tf.train.GradientDescentOptimizer()
         + `tf.compat.v1.train.GradientDescentOptimizer(learning_rate, use_locking=False, name='GradientDescent')`
            + `minimize(loss, global_step=None, var_list=None, gate_gradients=GATE_OP,aggregation_method=None, colocate_gradients_with_ops=False, name=None,grad_loss=None)`
               + 最小化loss，并且更新参数
               + loss，需要最小化的损失函数
               + global_step, 参数迭代更新的次数
               + var_list, 需要更新的参数，默认是拥有key为GraphKeys.TRAINABLE_VARIABLES的参数
      9. 初始化参数
         + tf.intialize_all_variables()
         ```
         init = tf.intialize_all_variables()
         sess.run(init)
         ```
      10. 评估模型
         + `tf.compat.v1.argmax(input, axis=None, name=None, dimension=None, output_type=tf.dtypes.int64)`
            + 返回输入tensor在某个方向上(axis)的最大值的索引
         + `tf.cast(x, dtype, name=None)` 张量的类型转换
         + `tf.math.equal(x, y, name=None)` 判断x,y两个张量的各个元素是否相等，返回Ture False数组
   4. 卷积神经网络+Softmax实现Mnist
      1. 生成正态分布的权重
         + `tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32)`
         + `tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None)`
      2. 卷积和池化
         + 卷积操作
            + `tf.compat.v1.nn.conv2d(input, filter=None, strides=None, padding=None, use_cudnn_on_gpu=True,data_format='NHWC', dilations=[1, 1, 1, 1], name=None, filters=None)`
               + 给定一个`[batch, in_height, in_width, in_channels]`的输入张量和一个`[filter_height, filter_width, in_channels, out_channels]`的过滤器/内核张量，这个op执行以下操作:
                  1. 使用shape将过滤器压扁为二维矩阵`[filter_height * filter_width * in_channels, output_channels]`
                  2. 从输入张量中提取图像块，形成一个形状的虚拟张量`[batch, out_height, out_width, filter_height * filter_width * in_channels]`
                  3. 对于每个图像块，将滤波矩阵和图像块向量右乘。
               + input: 输入的四维张量，四个维度的顺序决定于`data_format`
               + filter：四维张量，类型同input，`[filter_height, filter_width, in_channels, out_channels]`
               + strides:步长必须为`strides[0] = strides[3] = 1`，长度为1、2或4的整数或整数列表。输入的每个维度的滑动窗口的步长。如果给定一个值，它将被复制到H和W维中。默认情况下，N和C维被设置为1。维度顺序由data_format的值决定
               + padding: 字符串`SAME`或`VALID`指示要使用的填充算法的类型，或者一个列表指示每个维度的开始和结束处的显式填充。当使用显式填充且data_format为“NHWC”时，其格式应该是`[[0,0]，[pad_top, pad_bottom]， [pad_left, pad_right]，[0,0]]`。当使用显式内边距且data_format为“NCHW”时，其格式应为`[[0,0]，[0,0]，[pad_top, pad_bottom]， [pad_left, pad_right]]`
                  + "SAME": 卷积核的中心与图像重叠开始
                  + "VALID": 卷积核完全进入图像开始 
               + data_format: 字符串"NHWC", "NCHW", N代表batch， C代表channels
               + filters： 过滤器别名            
         + 池化操作
            + `tf.compat.v1.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None, input=None)`
               + 对输入进行最大卷积
               + value: 四维张量，形式由data_format决定
               + ksize: 长度为1、2或4的整数或整数列表, 输入张量每个维度的窗口大小。
               + strides: 长度为1、2或4的整数或整数列表, 输入张量每个维度的滑动窗口的步长。
               + padding: 'SAME', 'VALID'
               + data_format: 字符串。支持'NHWC'、'NCHW'和'NCHW_VECT_C'
      3. reshape 张量
         + `tf.reshape(tensor, shape, name=None)`
            + shape中如果有分量为-1, 表示计算该维度的大小，保持整体大小不变。如果是`[-1]`，则压缩为一维张量
            + 最多只能有一个维度为-1
      4. drop out 
         + `tf.nn.dropout(x, rate, noise_shape=None, seed=None, name=None)`
            + 以概率rate的情况下，x的元素被设为0。其余的elemenst按1.0 / (1 - rate)的比例放大，以保留最终输出期望值
            + noise_shape: 默认情况下，每个元素都是独立保存或删除的。如果指定了noise_shape，那么它会影响到x的形状，并且只有具有`noise_shape[i] == shape(x)[i]`的维度才能做出独立的保留或者删除。这对于从图像或序列中删除整个通道(整行或者整列)非常有用。
            ```
            >>> x = tf.ones([3,10]) 
            >>> tf.nn.dropout(x, rate = 2/3, noise_shape=[1,10]).numpy() 
            array([[0., 0., 0., 0., 3., 3., 3., 0., 3., 0.],
                   [0., 0., 0., 0., 3., 3., 3., 0., 3., 0.],
                   [0., 0., 0., 0., 3., 3., 3., 0., 3., 0.]], dtype=float32)
            ```
         
## Chapter 3 Tensorflow运作方式入门
   1. 上下文管理器 `tf.name_scope(name)`
      + 此上下文管理器将创建一个名称作用域，该作用域将使其内添加的所有操作的名称具有前缀。
      ```
      def my_op(a, b, c, name=None):
        with tf.name_scope("MyOp") as scope:
          a = tf.convert_to_tensor(a, name="a")
          b = tf.convert_to_tensor(b, name="b")
          c = tf.convert_to_tensor(c, name="c")
          # Define some computation that uses `a`, `b`, and `c`.
          return foo_op(..., name=scope)
          # 当程序执行时, 张量a, b, c将会拥有名称MyOp/a, MyOp/b, MyOp/c
      ```
   2. tf.expand_dims() 扩展张量的维度
      + `tf.expand_dims(input, axis, name=None)`
      + 给定一个张量输入，该operation在输入张量上所指定的`axis`维度索引上插入一个尺寸为1的维数。维度索引从零开始;如果`axis`指定一个负数，则从末尾开始向后计数。如果希望向单个元素添加批处理维度，则此操作非常有用。例如，如果您有一个shape `[height, width, channels]`的单一图像，您可以使用expand_dims(image, 0)使其成为一批图像，这将生成shape `[1, height, width, channels]`。
         + axis: 需要插入维度的所在的索引
      ```
      >>> t = [[1, 2, 3],[4, 5, 6]] # shape [2, 3] 
      >>> tf.expand_dims(t, 0)
      <tf.Tensor: shape=(1, 2, 3), dtype=int32, numpy=
      array([[[1, 2, 3],
            [4, 5, 6]]], dtype=int32)>
      >>> tf.expand_dims(t, 1)
      <tf.Tensor: shape=(2, 1, 3), dtype=int32, numpy=
      array([[[1, 2, 3]],
             [[4, 5, 6]]], dtype=int32)>
      >>> tf.expand_dims(t, 2)
      <tf.Tensor: shape=(2, 3, 1), dtype=int32, numpy=
      array([[[1],
              [2],
              [3]],
             [[4],
              [5],
              [6]]], dtype=int32)>
      ```
   3. tf.concat() 连接张量
      + `tf.concat(values, axis, name='concat')` 沿着某一个维度连接张量
      + 输入张量的维数必须匹配，除axis外的所有维数必须相等。
      ```
      >>> t1 = [[1, 2, 3], [4, 5, 6]] 
      >>> t2 = [[7, 8, 9], [10, 11, 12]] 
      >>> tf.concat([t1, t2], 0) 
      <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
      array([[ 1,  2,  3],
             [ 4,  5,  6],
             [ 7,  8,  9],
             [10, 11, 12]], dtype=int32)>
      >>> tf.concat([t1, t2], 1) 
      <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
      array([[ 1,  2,  3,  7,  8,  9],
             [ 4,  5,  6, 10, 11, 12]], dtype=int32)>
      ```
   4. 得到稀疏矩阵
      + `tf.compat.v1.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value=0, validate_indices=True, name=None)`
         + 将稀疏表示转换为稠密张量。
         ```
         import tensorflow.compat.v1 as tf

         indices = tf.reshape(tf.range(0, 10 ,1), [10, 1])
         labels=tf.expand_dims(tf.constant([0,2,3,6,7,9,1,3,5,4]),1)

         tf.sparse_to_dense(
               tf.concat(values=[indices, labels], axis=1),
               [10, 10], 1.0, 0.0)
         # <tf.Tensor: shape=(10, 10), dtype=float32, numpy=
         # array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          #       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
          #       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
          #       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
          #       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
          #       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
          #       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
          #       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
          #       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
          #       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]], dtype=float32)>
         ```
      + `tf.sparse.SparseTensor(indices, values, dense_shape)`
         + indices: 稀疏矩阵中需要设定值的索引
         + values: 稀疏矩阵中需要设定的值
         + dense_shape: 输出矩阵的维度
         ```
         tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
         # [[1, 0, 0, 0]
         # [0, 0, 2, 0]
         # [0, 0, 0, 0]]
         ```
      + `tf.sparse.to_dense(sp_input, default_value=None, validate_indices=True, name=None)`
         + sp_input: 输入的稀疏矩阵, 非设定值都为0
         + default_value: 为sp_input中未指定的索引设置填充值。默认值为零
         + validate_indices: 一个布尔值。如果为真，则检查索引以确保它们按字典顺序排序，并且没有重复。
         ```
         # m
         # [[0 a 0 b 0]
         # [0 0 0 0 0]
         # [c 0 0 0 0]]
         tf.sparse.to_dense(m, default_value=x)
         # [[x a x b x]
         # [x x x x x]
         # [c x x x x]]
         ```
   5. 保存检查点
      + tf.train.Saver() 存储或者加载变量
      + `tf.compat.v1.train.Saver(var_list=None, reshape=False, sharded=False, max_to_keep=5,keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False,saver_def=None, builder=None, defer_build=False, allow_empty=False,write_version=tf.train.SaverDef.V2, pad_step_number=False,save_relative_paths=False, filename=None)`
         + Checkpoints是私有格式的二进制文件，它将变量名映射到张量值
         + Saver可以自动编号Checkpoints文件名。这使在训练模型时可以在不同的步骤中保持多个Checkpoints。例如，您可以使用训练步骤编号给Checkpoints文件名编号。为了避免磁盘被填满，保护程序自动管理Checkpoints。例如，他们只能保存最近的N个文件，或者每N个小时的训练只能保存一个检查点。
         + 可以通过传递参数`global_step`来为checkpoints文件编号
         + `max_to_keep`: 指示要保留的最近检查点文件的最大数量, 默认为5
         + `keep_checkpoint_every_n_hours`: 每训练N小时，保存一次checkpoints，默认是10000h
            + var_list: 需要保存的参数列表，默认是所有可保存的变量
            
         + Saver需要调用save函数才可以保存文件
            + `save(sess, save_path, global_step=None, latest_filename=None,meta_graph_suffix='meta', write_meta_graph=True, write_state=True,strip_default_attrs=False, save_debug_info=False)`
               + sess: 用来保存变量的会话
               + save_path: 保存路径
               + global_step: 如果提供，则用来编号checkpoint的文件名
               + latest_filename: protocol buffer文件的可选名称，该文件将包含最近的checkpoint列表。该文件与checkpoint文件保存在同一个目录中，由保护程序自动管理，以跟踪最近的检查点。
               + meta_graph_suffix: MetaGraphDef文件的后缀
               + write_meta_graph: 布尔值，是否存储meta graph文件
         ```
         saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
         saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
         ```
         + restore加载文件
            + `restore(sess, save_path)`
   6. tf.math.in_top_k()
      + `tf.math.in_top_k(targets, predictions, k, name=None)` 检查目标是否在预测值的前k个里面
         + predictions: 预测的结果，预测矩阵大小为样本数×标注的label类的个数的二维矩阵。
         + targets: 实际的标签，大小为样本数。
         + k: 每个样本的预测结果的前k个最大的数的标签里面是否包含targets中的标签，一般都是取1，即取预测最大概率的索引与标签对比。
         ```
         import tensorflow as tf
 
         logits = tf.Variable(tf.truncated_normal(shape=[10,5],stddev=1.0))
         labels = tf.constant([0,0,0,0,0,0,0,0,0,0])
 
         top_1_op = tf.nn.in_top_k(logits,labels,1)
         top_2_op = tf.nn.in_top_k(logits,labels,2)
 
         with tf.Session() as sess:
             sess.run(tf.global_variables_initializer())
             print(logits.eval())
             print(labels.eval())
             print(top_1_op.eval())
             print(top_2_op.eval())
         # [[-0.01835343 -1.68495178 -0.67901242 -0.20486258 -0.22725371]
         # [ 1.84425163 -1.25509632  0.07132829 -1.81082523 -0.44123012]
         # [-0.4354656   0.1805554   0.81912154  0.04202025 -1.99823892]
         # [ 0.53393573  0.91522688 -1.88455033 -0.44571343  0.07805539]
         # [ 0.01253182  0.16593859  0.0918197   0.8079409   0.13442524]
         # [ 0.08205117 -0.26857412  0.02542082  0.38249066 -0.01555154]
         # [-1.02280331  0.18952899  0.49389341  0.58559865  0.80859423]
         # [ 0.35019293 -1.17765355  0.66553122  1.91787696  0.5998978 ]
         # [ 0.81723028  0.92895705  0.86031818  1.57651412  0.94040418]
         # [-0.83766556 -1.75260925  0.13499574 -0.06683849 -0.99427927]]
         # [0 0 0 0 0 0 0 0 0 0]
         # [ True  True False False False False False False False False]
         # [ True  True False  True False  True False False False False]
         ```
 
 
# Tensorflow2.0 英文官方文档
## Chapter 1 Keras机器学习基础知识
   1. 基本图像分类
   2. 使用TF Hub进行文本分类