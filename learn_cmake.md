# cmake 教程
## 简介
cmake是一个跨平台的makefile生成工具，可以为项目自动生成makefile，其亮点在于编译复杂项目上的应用
## 单个文件编译
假设只有一个`helloworld.cpp`，则最基本的CMakeList.txt如下
```cmake
cmake_minimum_required(VERSION 2.8.9)
project(hello)
add_executable(hello helloworld.cpp)
```
+ 第一行用于指定cmake最低版本
+ 第二行指定项目名称 该名称任意
+ 第三行指定编译一个可执行文件，`hello`表示生成可执行文件的文件名，也是任意的，`helloworld.cpp`指定源文件
## 包含目录结构的项目
项目结构：
```
│  CMakeLists.txt
│  
├─include
│      Student.h
│      
└─src
        mainapp.cpp
        Student.cpp
```
此时，CMakeLists.txt如下：
```cmake
cmake_minimum_required(VERSION 2.8.9)
project(test)
# Bring the headers, such as Student.h into the project
include_directories(include)

# Can manually add the sources using the set command as follows:
# set(SOURCES src/mainapp.cpp src/Student.cpp)

# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

add_executable(testStudent ${SOURCES})
```
+ `include_directories()`包含头文件目录
+ 使用`set(SOURCES ...)`或`GLOB(or GLOB_RECURSE)`设置源文件SOURCES
+ `add_executable`使用变量SOURCES，代替具体的文件名
## 动态库的编译(.so)
项目结构：
```
│  CMakeLists.txt
│  
├─build
├─include
│      Student.h
│      
└─src
        Student.cpp
```
CMakeLists.txt如下：
```cmake
cmake_minimum_required(VERSION 2.8.9)
project(directory_test)
set(CMAKE_BUILD_TYPE Release)

#Bring the headers, such as Student.h into the project
include_directories(include)

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

#Generate the shared library from the sources
add_library(testStudent SHARED ${SOURCES})

# Set the location for library installation -- i.e., /usr/lib in this case
# not really necessary in this example. Use "sudo make install" to apply
install(TARGETS testStudent DESTINATION /usr/lib)
```
+ 使用`add_library()`以及参数`SHARED`
+ `install`指定安装目录，执行`sudo make install`时动态库将被安装在/usr/lib目录下
## 静态库编译(.a)
项目结构：
```
│  CMakeLists.txt
│  
├─build
├─include
│      Student.h
│      
└─src
        Student.cpp
```
CMakeLists.txt如下：
```cmake
cmake_minimum_required(VERSION 2.8.9)
project(directory_test)
set(CMAKE_BUILD_TYPE Release)

#Bring the headers, such as Student.h into the project
include_directories(include)

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

#Generate the static library from the sources
add_library(testStudent STATIC ${SOURCES})

#Set the location for library installation -- i.e., /usr/lib in this case
# not really necessary in this example. Use "sudo make install" to apply
install(TARGETS testStudent DESTINATION /usr/lib)
```
+ 只需要将`add_library()`中的 `SHARED`改为`STATIC`
## 使用静态库或动态库
CMakeLists.txt如下：
```cmake
cmake_minimum_required(VERSION 2.8.9)
project (TestLibrary)

#For the shared library:
set ( PROJECT_LINK_LIBS libtestStudent.so )
link_directories( ~/exploringBB/extras/cmake/studentlib_shared/build )

#For the static library:
#set ( PROJECT_LINK_LIBS libtestStudent.a )
#link_directories( ~/exploringBB/extras/cmake/studentlib_static/build )

include_directories(~/exploringBB/extras/cmake/studentlib_shared/include)

add_executable(libtest libtest.cpp)
target_link_libraries(libtest ${PROJECT_LINK_LIBS} )
```
# cmake command line
## find_package() 函数
可以帮助自动寻找相关依赖库的路径
```cmake
find_package(dependence_name REQUIRED)
# 寻找依赖的相关头文件和库文件，如果找到了则
# dependence_name_FOUND - 如果找到库，则设置为true，否则为false
# dependence_name_INCLUDE_DIRS或<NAME> _INCLUDES - 包导出的包含路径
# dependence_name_LIBRARIES或<NAME> _LIBS - 由包导出的库
```
+ ***version和EXACT***：version指的是版本，如果指定就必须检查找到的包的版本是否和version兼容。如果指定EXACT则表示必须完全匹配的版本而不是兼容版本就可以
+ ***QUIET***：表示如果查找失败，不会在屏幕进行输出（但是如果指定了REQUIRED字段，则QUIET无效，仍然会输出查找失败提示语）
+ ***MODULE***：默认“如果Module模式查找失败则回退到Config模式进行查找”，但是假如设定了MODULE选项，那么就只在Module模式查找，如果Module模式下查找失败并不回落到Config模式查找。
+ ***CONFIG***：直接在Config模式中查找
+ ***REQUIRED***：表示一定要找到包，找不到的话就立即停掉整个cmake。而如果不指定REQUIRED则cmake会继续执行。
+ ***COMPONENTS***：表示查找的包中必须要找到的组件(components)，如果有任何一个找不到就算失败，类似于REQUIRED，导致cmake停止执行。
+ ***OPTIONAL_COMPONENTS***：可选的模块，找不到也不会让cmake停止执行
### find_package()原理
首先，cmake本身不提供任何搜索库的便捷方法，所有搜索库并给变量赋值的操作必须由cmake代码完成，比如FindXXX.cmake和XXXConfig.cmake。只不过，库的作者通常会提供这两个文件，以方便使用者调用。
find_package采用两种模式搜索库：
+ Module模式：搜索CMAKE_MODULE_PATH指定路径下的FindXXX.cmake文件，执行该文件从而找到XXX库。其中，具体查找库并给XXX_INCLUDE_DIRS和XXX_LIBRARIES两个变量赋值的操作由FindXXX.cmake模块完成（先搜索当前项目里面的Module文件夹里面提供的FindXXX.cmake，然后再搜索系统路径/usr/local/share/cmake-x.y/Modules/FindXXX.cmake）
+ Config模式：搜索XXX_DIR指定路径下的XXXConfig.cmake文件，执行该文件从而找到XXX库。其中具体查找库并给XXX_INCLUDE_DIRS和XXX_LIBRARIES两个变量赋值的操作由XXXConfig.cmake模块完成。
## CMAKE_BUILD_TYPE
设置编译模式，一般为`Debug`和`Release`两种模式
+ 默认模式为`Debug`
