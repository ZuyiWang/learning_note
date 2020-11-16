# LateX论文编辑笔记 #
### 学会查看官方文档 ###
+ 打开cmd，输入`texdoc + 查看文档名称` eg: `texdoc lshort-zh %查看中文简版文档`
## 1.  源文件的基本结构 ##
+ 导言区
```latex
\documentclass{article} %book,report, letter
\title{My first Document}
\author{ZuyiWang}
\date{\today}
```
+ 正文区（文稿区）
```latex
begin{document}   %只能有一个
	\maketitle	 %生成标题
	%空行代表换行 多个空行识别为一个空行
	Hello World!	
	$f(x)$   %行内数学模式
	$$f(x)=x^2$$  %行间数学模式
end{document}
```
## 2. 中文处理办法 ##
+ **设置——构建——默认编译器——XeLaTeX**
+ **设置——编辑器——默认格式——UTF-8**
+ `\usepackage{ctex}`
+ 修改字体：\heiti, \kaishu 等等，放在中文之前 eg：`\title{\heiti 勾股定理}`
```latex
\begin{equation}  %产生带编号的公式
a^2 = b^2 +c^2
\end{equation}
```
+ 查看文档 cmd中输入`texdoc ctex`
## 3.  字体字号设置 ##
+ 字体族设置 (罗马字体 无衬线字体 打字机字体)
```latex
\textrm{Roman Family} \textsf{Sans Serif Family} \texttt{Typewriter Family}

\rmfamily Roman Family  
\sffamily Sans Serif Family
\ttfamily Typewriter Family
```
+ 字体系列设置 (粗细 宽度)
```latex
\textmd{Medium Series} \textbf{Boldface Series} 

\mdseries Roman Family  
\bfseries Boldface Series
```
+ 字体形状设置 （直立 斜体 伪斜体 小型大写）
```latex
\textup{Upright Shape} \textit{Italic Shape} 
\textsl{Slanted Shape} \textsc{Small Caps Shape} 

\upshape Upright Shape
\itshape Italic Shape
\slshape Slanted Shape
\scshape Small Caps Shape
```
+中文字体设置 
```latex
{\songti 宋体} {\heiti 黑体} {\fangsong 仿宋} {\kaishu 楷书}
```
+字体大小设置 
```latex
{\tiny Hello} 
{\scriptsize Hello} 
{\footnotesize Hello}
{\small Hello}
{\normalsize Hello}
{\large Hello}
{\Large Hello}
{\LARGE Hello}
{\huge Hello}
{\Huge Hello}
```
## 4.  文档的基本结构 ##
## 5.  特殊字符 ##
## 6.  插图 ##
## 7.  表格 ##
## 8.  浮动体 ##
## 9.  数学公式 ##
## 10.  参考文献 ##
## 11.  自定义 ##
## 12.  插入代码 ##
