# Python 完整语法笔记

## 目录
1. [基础语法](#一基础语法)
2. [数据结构](#二数据结构)
3. [控制流](#三控制流)
4. [函数](#四函数)
5. [面向对象](#五面向对象)
6. [模块与包](#六模块与包)
7. [异常处理](#七异常处理)
8. [文件操作](#八文件操作)
9. [高级特性](#九高级特性)
10. [容器操作详解](#十容器操作详解)
11. [并发编程](#十一并发编程)
12. [推导式](#十二推导式)
13. [解包与序列操作](#十三解包与序列操作)
14. [高级函数特性](#十四高级函数特性)
15. [更多高级语法](#十五更多高级语法)
16. [常用内置函数](#十六常用内置函数)
17. [常用标准库](#十七常用标准库)
18. [实用技巧与代码片段](#十八实用技巧与代码片段)
19. [最佳实践与代码风格 (PEP 8)](#十九最佳实践与代码风格-pep-8)
20. [常见陷阱与误区](#二十常见陷阱与误区)
21. [网络编程](#二十一网络编程)
22. [数据处理与序列化](#二十二数据处理与序列化)
23. [数据库操作](#二十三数据库操作)
---

## 一、基础语法

### 变量与数据类型
```python
# 基本类型
x = 10          # int 整数
y = 3.14        # float 浮点数
name = "Alice"  # str 字符串
flag = True     # bool 布尔值
nothing = None  # NoneType 空值

# 类型转换
int("10")       # 字符串转整数
str(10)         # 整数转字符串
float("3.14")   # 字符串转浮点数
```

### 运算符
```python
# 算术运算符
+    # 加
-    # 减
*    # 乘
/    # 除（浮点除法）
//   # 整除
%    # 取余
**   # 幂运算

# 比较运算符
==   # 等于
!=   # 不等于
>    # 大于
<    # 小于
>=   # 大于等于
<=   # 小于等于

# 逻辑运算符
and  # 与
or   # 或
not  # 非

# 身份运算符
is        # 是否是同一对象
is not    # 是否不是同一对象

# 成员运算符
in        # 是否在容器中
not in    # 是否不在容器中
```

### 注释
```python
# 单行注释

"""
多行注释
可以跨越多行
"""

'''
也可以用单引号
'''
```

---

## 二、数据结构

### 列表 (List) - 可变有序
```python
# 创建
lst = [1, 2, 3, 4, 5]
empty = []
mixed = [1, "hello", 3.14, True]

# 访问
lst[0]      # 第一个元素
lst[-1]     # 最后一个元素
lst[1:3]    # 切片 [1, 2]

# 修改
lst[0] = 10
lst.append(6)           # 末尾添加
lst.insert(0, 0)        # 指定位置插入
lst.extend([7, 8])      # 扩展列表
lst.remove(3)           # 删除值为3的元素
lst.pop()               # 删除末尾元素
lst.pop(0)              # 删除指定位置

# 查询
len(lst)                # 长度
3 in lst                # 是否存在
lst.count(2)            # 统计出现次数
lst.index(5)            # 查找索引

# 排序
lst.sort()              # 原地排序
sorted(lst)             # 返回新列表
lst.reverse()           # 反转
```

### 元组 (Tuple) - 不可变有序
```python
# 创建
tup = (1, 2, 3)
single = (1,)           # 单元素元组需要逗号
no_parens = 1, 2, 3     # 可省略括号

# 访问（同列表）
tup[0]
tup[1:3]

# 解包
a, b, c = (1, 2, 3)
```

### 字典 (Dictionary) - 键值对
```python
# 创建
dic = {'name': 'Alice', 'age': 25}
dic = dict(name='Alice', age=25)

# 访问
dic['name']             # 直接访问
dic.get('name')         # 安全访问
dic.get('height', 170)  # 带默认值

# 修改
dic['age'] = 26
dic.update({'city': 'Beijing'})

# 删除
del dic['age']
dic.pop('name')
dic.clear()

# 遍历
for key in dic.keys():
    pass
for value in dic.values():
    pass
for key, value in dic.items():
    pass
```

### 集合 (Set) - 无序不重复
```python
# 创建
s = {1, 2, 3}
s = set([1, 2, 2, 3])   # 自动去重

# 操作
s.add(4)
s.remove(3)
s.discard(5)            # 不存在不报错

# 集合运算
s1 | s2                 # 并集
s1 & s2                 # 交集
s1 - s2                 # 差集
s1 ^ s2                 # 对称差集
```

---

## 三、控制流

### 条件语句
```python
if condition:
    # 代码块
    pass
elif another_condition:
    pass
else:
    pass

# 三元表达式
result = value1 if condition else value2
```

### 循环
```python
# for 循环
for item in iterable:
    print(item)

for i in range(10):
    print(i)

# while 循环
while condition:
    # 代码块
    pass

# 控制语句
break       # 跳出循环
continue    # 跳过本次迭代

# else 子句（循环正常结束时执行）
for i in range(5):
    if i == 3:
        break
else:
    print("循环正常结束")
```

### 列表推导式
```python
[x**2 for x in range(10)]
[x for x in range(10) if x % 2 == 0]
[(x, y) for x in range(3) for y in range(3)]
```

---

## 四、函数

### 定义与调用
```python
def function_name(param1, param2=default):
    """文档字符串"""
    # 函数体
    return value

# 调用
result = function_name(arg1, arg2)
```

### 参数类型
```python
# 位置参数
def func(a, b):
    pass

# 默认参数
def func(a, b=10):
    pass

# 可变参数
def func(*args):        # 元组
    pass

def func(**kwargs):     # 字典
    pass

# 组合使用
def func(a, b=10, *args, **kwargs):
    pass

# 强制关键字参数（Python 3+）
def func(a, *, b, c):
    pass
func(1, b=2, c=3)       # b和c必须用关键字
```

### Lambda 表达式
```python
lambda x: x * 2
lambda x, y: x + y

# 常用于高阶函数
sorted(lst, key=lambda x: x[1])
map(lambda x: x**2, range(10))
filter(lambda x: x % 2 == 0, range(10))
```

### 装饰器
```python
def decorator(func):
    def wrapper(*args, **kwargs):
        # 前置处理
        result = func(*args, **kwargs)
        # 后置处理
        return result
    return wrapper

@decorator
def my_function():
    pass
```

---

## 五、面向对象

### 类定义
```python
class ClassName:
    # 类属性
    class_var = "shared"
    
    def __init__(self, param):
        # 实例属性
        self.instance_var = param
    
    # 实例方法
    def method(self):
        return self.instance_var
    
    # 类方法
    @classmethod
    def class_method(cls):
        return cls.class_var
    
    # 静态方法
    @staticmethod
    def static_method():
        return "static"
```

### 继承
```python
class Parent:
    def parent_method(self):
        pass

class Child(Parent):
    def __init__(self):
        super().__init__()  # 调用父类构造函数
    
    def child_method(self):
        pass

# 多重继承
class Child(Parent1, Parent2):
    pass
```

### 特殊方法
```python
class MyClass:
    def __init__(self):         # 构造函数
        pass
    
    def __str__(self):          # str() 和 print()
        return "string"
    
    def __repr__(self):         # repr() 和交互式显示
        return "representation"
    
    def __len__(self):          # len()
        return 0
    
    def __getitem__(self, key): # obj[key]
        pass
    
    def __setitem__(self, key, value):  # obj[key] = value
        pass
    
    def __add__(self, other):   # +
        pass
    
    def __eq__(self, other):    # ==
        pass
```

---

## 六、模块与包

### 导入
```python
import module
from module import function
from module import *
import module as alias
from package.module import function
```

### 创建模块
```python
# mymodule.py
def my_function():
    pass

# 使用
import mymodule
mymodule.my_function()

# 判断是否为主程序
if __name__ == '__main__':
    # 只在直接运行时执行
    pass
```

---

## 七、异常处理

```python
try:
    # 可能出错的代码
    risky_operation()
except SpecificError as e:
    # 处理特定异常
    handle_error(e)
except (Error1, Error2):
    # 处理多种异常
    pass
except:
    # 处理所有异常
    pass
else:
    # 没有异常时执行
    pass
finally:
    # 无论如何都执行
    cleanup()

# 抛出异常
raise ValueError("错误信息")

# 自定义异常
class MyError(Exception):
    pass
```

---

## 八、文件操作

```python
# 读取文件
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()          # 读取全部
    line = f.readline()         # 读取一行
    lines = f.readlines()       # 读取所有行

# 写入文件
with open('file.txt', 'w') as f:
    f.write("content")
    f.writelines(["line1\n", "line2\n"])

# 追加模式
with open('file.txt', 'a') as f:
    f.write("append")

# 二进制模式
with open('file.bin', 'rb') as f:
    data = f.read()

# 文件模式
# 'r'  - 读取（默认）
# 'w'  - 写入（覆盖）
# 'a'  - 追加
# 'b'  - 二进制模式
# 'r+' - 读写
```

---

## 九、高级特性

### 生成器
```python
# 生成器函数
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()
next(gen)  # 1

# 生成器表达式
gen = (x**2 for x in range(10))
```

### 迭代器
```python
class MyIterator:
    def __iter__(self):
        return self
    
    def __next__(self):
        # 返回下一个值
        # 没有值时抛出 StopIteration
        pass

iter_obj = iter(iterable)
next(iter_obj)
```

### 上下文管理器
```python
class MyContext:
    def __enter__(self):
        # 进入时执行
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出时执行
        return False

with MyContext() as ctx:
    # 使用上下文
    pass
```

---

## 十、容器操作详解

### 列表 (List) 完整操作

#### 创建
```python
lst = [1, 2, 3]
lst = list(range(5))
lst = [x**2 for x in range(5)]  # 推导式
```

#### 访问与切片
```python
lst[0]          # 索引访问
lst[-1]         # 倒数第一个
lst[1:4]        # 切片 [起始:结束:步长]
lst[::2]        # 每隔一个
lst[::-1]       # 反转
```

#### 增加元素
```python
lst.append(x)           # 末尾添加
lst.extend([a, b])      # 添加多个
lst.insert(index, x)    # 指定位置插入
lst += [a, b]           # 连接
```

#### 删除元素
```python
lst.remove(x)           # 删除第一个x
lst.pop()               # 删除末尾
lst.pop(index)          # 删除指定位置
lst.clear()             # 清空
del lst[index]          # 删除
del lst[1:3]            # 删除切片
```

#### 查询与排序
```python
x in lst
lst.index(x)
lst.count(x)
len(lst)
lst.sort()              # 原地排序
sorted(lst)             # 返回新列表
lst.reverse()           # 反转
```

### 字典 (Dictionary) 完整操作

#### 创建
```python
dic = {'a': 1, 'b': 2}
dic = dict(a=1, b=2)
dic = {k: v for k, v in pairs}  # 推导式
```

#### 访问与修改
```python
dic['key']
dic.get('key', default)
dic['key'] = value
dic.update({'c': 3})
dic.setdefault('key', default)
```

#### 删除
```python
del dic['key']
dic.pop('key')
dic.popitem()           # 删除最后一个
dic.clear()
```

#### 遍历
```python
for key in dic:
for key in dic.keys():
for value in dic.values():
for key, value in dic.items():
```

### 集合 (Set) 完整操作

#### 创建
```python
s = {1, 2, 3}
s = set([1, 2, 3])
s = {x for x in range(5)}  # 推导式
```

#### 基本操作
```python
s.add(x)
s.remove(x)             # 不存在会报错
s.discard(x)            # 不存在不报错
s.pop()                 # 随机删除
s.clear()
```

#### 集合运算
```python
s1 | s2                 # 并集
s1 & s2                 # 交集
s1 - s2                 # 差集
s1 ^ s2                 # 对称差集
s1 <= s2                # 子集
s1 >= s2                # 超集
```

### 字符串 (String) 操作

#### 查找
```python
s.find('sub')           # 返回索引，未找到返回-1
s.index('sub')          # 返回索引，未找到报错
s.count('sub')
'sub' in s
s.startswith('pre')
s.endswith('suf')
```

#### 转换
```python
s.upper()
s.lower()
s.capitalize()          # 首字母大写
s.title()               # 每个单词首字母大写
```

#### 分割与连接
```python
s.split(',')
s.rsplit(',', maxsplit=1)
s.splitlines()
','.join(list)
```

#### 修剪与替换
```python
s.strip()               # 去除两端空白
s.lstrip()
s.rstrip()
s.replace('old', 'new')
```

#### 格式化
```python
"Hello {}".format(name)
f"Hello {name}"         # f-string
f"{value:.2f}"          # 格式控制
```

---

## 十一、并发编程

### 多线程 (Threading)

#### 基本使用
```python
import threading
import time

def worker(name, delay):
    print(f"线程 {name} 开始")
    time.sleep(delay)
    print(f"线程 {name} 完成")

# 创建线程
t1 = threading.Thread(target=worker, args=("A", 2))
t2 = threading.Thread(target=worker, args=("B", 1))

# 启动和等待
t1.start()
t2.start()
t1.join()
t2.join()
```

#### 线程同步
```python
# Lock（互斥锁）
lock = threading.Lock()

def increment():
    with lock:
        # 临界区
        counter += 1

# RLock（可重入锁）
rlock = threading.RLock()

# Semaphore（信号量）
semaphore = threading.Semaphore(3)

with semaphore:
    # 最多3个线程同时访问
    pass

# Event（事件）
event = threading.Event()
event.wait()            # 等待事件
event.set()             # 触发事件

# Condition（条件变量）
condition = threading.Condition()

with condition:
    condition.wait()    # 等待通知
    condition.notify()  # 通知等待的线程
```

#### 线程池
```python
from concurrent.futures import ThreadPoolExecutor

def task(n):
    return n * n

with ThreadPoolExecutor(max_workers=5) as executor:
    # submit 方式
    futures = [executor.submit(task, i) for i in range(10)]
    results = [f.result() for f in futures]
    
    # map 方式
    results = list(executor.map(task, range(10)))
```

### 多进程 (Multiprocessing)

#### 基本使用
```python
import multiprocessing

def worker(name):
    print(f"进程 {name} 开始")
    return name

if __name__ == '__main__':
    p1 = multiprocessing.Process(target=worker, args=("A",))
    p2 = multiprocessing.Process(target=worker, args=("B",))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
```

#### 进程池
```python
from multiprocessing import Pool

def task(n):
    return n * n

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        results = pool.map(task, range(10))
```

#### 进程间通信
```python
from multiprocessing import Queue, Pipe, Manager

# Queue（队列）
queue = Queue()
queue.put(item)
item = queue.get()

# Pipe（管道）
parent_conn, child_conn = Pipe()
parent_conn.send(data)
data = child_conn.recv()

# Manager（共享状态）
with Manager() as manager:
    shared_dict = manager.dict()
    shared_list = manager.list()
```

### 协程 (Asyncio)

#### 基本使用
```python
import asyncio

async def task(name, delay):
    print(f"任务 {name} 开始")
    await asyncio.sleep(delay)
    print(f"任务 {name} 完成")
    return f"结果-{name}"

# 运行协程
asyncio.run(task("A", 1))
```

#### 并发执行
```python
async def main():
    # gather - 并发执行
    results = await asyncio.gather(
        task("A", 2),
        task("B", 1),
        task("C", 3)
    )
    
    # create_task
    task1 = asyncio.create_task(task("D", 1))
    task2 = asyncio.create_task(task("E", 2))
    
    result1 = await task1
    result2 = await task2

asyncio.run(main())
```

#### 异步上下文管理器
```python
class AsyncResource:
    async def __aenter__(self):
        print("获取资源")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("释放资源")

async def main():
    async with AsyncResource() as resource:
        # 使用资源
        pass
```

#### 异步队列
```python
async def producer(queue):
    for i in range(5):
        await queue.put(i)
    await queue.put(None)

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"消费: {item}")
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    await asyncio.gather(
        producer(queue),
        consumer(queue)
    )
```

### 并发方案选择

| 场景 | 方案 | 原因 |
|------|------|------|
| CPU密集型 | 多进程 | 绕过GIL，真正并行 |
| I/O密集型（同步API） | 多线程 | 简单易用 |
| I/O密集型（异步API） | 协程 | 高效轻量 |
| 混合任务 | 组合使用 | 发挥各自优势 |

---

## 十二、推导式

### 列表推导式
```python
# 基本形式
[x**2 for x in range(10)]

# 带条件
[x for x in range(10) if x % 2 == 0]

# 嵌套
[(x, y) for x in range(3) for y in range(3)]

# 多重条件
[x for x in range(100) if x % 2 == 0 if x % 5 == 0]
```

### 字典推导式
```python
{x: x**2 for x in range(5)}

# 从列表创建
keys = ['a', 'b', 'c']
values = [1, 2, 3]
{k: v for k, v in zip(keys, values)}

# 带条件
{x: x**2 for x in range(10) if x % 2 == 0}
```

### 集合推导式
```python
{x**2 for x in range(10)}
{x % 3 for x in range(10)}  # 自动去重
```

### 生成器表达式
```python
gen = (x**2 for x in range(1000000))  # 惰性求值
list(gen)  # 需要时才计算

# 直接用于函数
sum(x**2 for x in range(100))
```

---

## 十三、解包与序列操作

### 基本解包
```python
# 元组解包
a, b, c = (1, 2, 3)
a, b = b, a  # 交换值

# 忽略某些值
a, _, c = (1, 2, 3)
```

### 星号解包
```python
# * 收集剩余元素
a, *b, c = [1, 2, 3, 4, 5]  # a=1, b=[2,3,4], c=5

# * 解包列表
nums = [1, 2, 3]
print(*nums)  # 相当于 print(1, 2, 3)

# ** 解包字典
def func(a, b, c):
    return a + b + c

d = {'a': 1, 'b': 2, 'c': 3}
func(**d)

# 合并容器
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = [*list1, *list2]

dict1 = {'a': 1}
dict2 = {'b': 2}
combined = {**dict1, **dict2}
```

### 函数参数解包
```python
def func(a, b, *args, **kwargs):
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")

func(1, 2, 3, 4, x=5, y=6)
# a=1, b=2
# args=(3, 4)
# kwargs={'x': 5, 'y': 6}
```

---

## 十四、高级函数特性

### 闭包
```python
def outer(x):
    def inner(y):
        return x + y
    return inner

add_5 = outer(5)
print(add_5(3))  # 8
```

### 装饰器详解
```python
# 基本装饰器
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"耗时: {time.time() - start:.2f}秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)

# 带参数的装饰器
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")

# 保留原函数信息
from functools import wraps

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

### 偏函数
```python
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)
```

### 函数缓存
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

---

## 十五、更多高级语法

### 属性装饰器
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("半径不能为负")
        self._radius = value
    
    @property
    def area(self):
        return 3.14 * self._radius ** 2
```

### 数据类
```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    hobbies: list = None
```

### 类型注解
```python
from typing import List, Dict, Optional, Union

def process(items: List[int]) -> Dict[str, int]:
    return {"total": sum(items)}

def find_user(id: int) -> Optional[str]:
    return "Alice" if id == 1 else None
```

### 枚举
```python
from enum import Enum, auto

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

class Status(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
```

### 命名元组
```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)
```

### 上下文管理器
```python
from contextlib import contextmanager

@contextmanager
def my_context():
    print("进入")
    try:
        yield "资源"
    finally:
        print("退出")

with my_context() as resource:
    print(f"使用 {resource}")
```

### 生成器进阶
```python
# yield from
def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

# 双向通信
def echo_generator():
    while True:
        received = yield
        print(f"收到: {received}")

gen = echo_generator()
next(gen)
gen.send("Hello")
```

### 描述符
```python
class TypedProperty:
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"期望 {self.expected_type}")
        obj.__dict__[self.name] = value
```

### 模式匹配（Python 3.10+）
```python
def http_status(status):
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:
            return "Unknown"

# 模式匹配复杂结构
def process_command(command):
    match command.split():
        case ["quit"]:
            return "退出"
        case ["load", filename]:
            return f"加载 {filename}"
        case ["save", filename]:
            return f"保存 {filename}"
        case _:
            return "未知命令"

# 字典模式
def process_data(data):
    match data:
        case {"type": "user", "name": name, "age": age}:
            return f"用户: {name}, {age}岁"
        case {"type": "admin", **rest}:
            return f"管理员: {rest}"
        case _:
            return "未知数据"
```

### 海象运算符（:=）Python 3.8+
```python
# 在表达式中赋值
if (n := len(items)) > 10:
    print(f"列表太长: {n} 个元素")

# 在列表推导式中
[y for x in data if (y := process(x)) is not None]

# 简化循环
while (line := file.readline()) != "":
    process(line)
```

---

## 十六、常用内置函数

### 数学函数
```python
abs(-5)                 # 绝对值: 5
round(3.7)              # 四舍五入: 4
round(3.14159, 2)       # 保留2位: 3.14
pow(2, 3)               # 幂运算: 8
divmod(10, 3)           # 商和余数: (3, 1)
max(1, 2, 3)            # 最大值: 3
min([1, 2, 3])          # 最小值: 1
sum([1, 2, 3])          # 求和: 6
```

### 类型转换
```python
int("10")               # 字符串转整数
float("3.14")           # 字符串转浮点数
str(10)                 # 转字符串
bool(0)                 # 转布尔值: False
list("abc")             # ['a', 'b', 'c']
tuple([1, 2, 3])        # (1, 2, 3)
set([1, 2, 2, 3])       # {1, 2, 3}
dict([('a', 1)])        # {'a': 1}
```

### 序列操作
```python
len([1, 2, 3])          # 长度: 3
range(5)                # 0到4的范围
range(1, 10, 2)         # 1到9，步长2
enumerate(['a', 'b'])   # [(0,'a'), (1,'b')]
zip([1, 2], ['a', 'b']) # [(1,'a'), (2,'b')]
reversed([1, 2, 3])     # 反转迭代器
sorted([3, 1, 2])       # 排序: [1, 2, 3]
```

### 高阶函数
```python
map(func, iterable)     # 映射
filter(func, iterable)  # 过滤
reduce(func, iterable)  # 归约（需导入）

# 示例
list(map(lambda x: x**2, [1, 2, 3]))  # [1, 4, 9]
list(filter(lambda x: x > 0, [-1, 0, 1]))  # [1]

from functools import reduce
reduce(lambda x, y: x + y, [1, 2, 3, 4])  # 10
```

### 对象操作
```python
type(obj)               # 获取类型
isinstance(obj, type)   # 类型检查
id(obj)                 # 对象ID
dir(obj)                # 对象属性列表
hasattr(obj, 'attr')    # 是否有属性
getattr(obj, 'attr')    # 获取属性
setattr(obj, 'attr', val) # 设置属性
delattr(obj, 'attr')    # 删除属性
callable(obj)           # 是否可调用
```

### 输入输出
```python
print(*values, sep=' ', end='\n')
input(prompt)           # 获取用户输入
```

### 其他
```python
help(obj)               # 帮助文档
eval("1 + 2")           # 执行表达式: 3
exec("x = 1")           # 执行代码
compile(code, '<string>', 'exec')  # 编译代码
globals()               # 全局变量字典
locals()                # 局部变量字典
vars(obj)               # 对象的__dict__
```

---

## 十七、常用标准库

### collections - 容器数据类型
```python
from collections import (
    Counter,        # 计数器
    defaultdict,    # 默认字典
    OrderedDict,    # 有序字典
    deque,          # 双端队列
    namedtuple,     # 命名元组
    ChainMap        # 链式字典
)

# Counter
c = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
c.most_common(2)        # [('a', 3), ('b', 2)]

# defaultdict
d = defaultdict(list)
d['key'].append(1)      # 自动创建空列表

# deque
dq = deque([1, 2, 3])
dq.appendleft(0)        # 左侧添加
dq.pop()                # 右侧删除
```

### itertools - 迭代工具
```python
from itertools import (
    count,          # 无限计数
    cycle,          # 循环迭代
    repeat,         # 重复元素
    chain,          # 连接迭代器
    combinations,   # 组合
    permutations,   # 排列
    product,        # 笛卡尔积
    groupby,        # 分组
    islice,         # 切片
    takewhile,      # 条件获取
    dropwhile       # 条件跳过
)

# 示例
list(combinations([1, 2, 3], 2))  # [(1,2), (1,3), (2,3)]
list(permutations([1, 2, 3], 2))  # [(1,2), (1,3), (2,1), ...]
list(product([1, 2], ['a', 'b'])) # [(1,'a'), (1,'b'), (2,'a'), (2,'b')]
```

### datetime - 日期时间
```python
from datetime import datetime, date, time, timedelta

# 当前时间
now = datetime.now()
today = date.today()

# 创建时间
dt = datetime(2024, 1, 1, 12, 30, 0)
d = date(2024, 1, 1)
t = time(12, 30, 0)

# 格式化
now.strftime("%Y-%m-%d %H:%M:%S")

# 解析
datetime.strptime("2024-01-01", "%Y-%m-%d")

# 时间差
delta = timedelta(days=7, hours=2)
future = now + delta
```

### pathlib - 路径操作
```python
from pathlib import Path

# 创建路径
p = Path('folder/file.txt')
p = Path.home() / 'documents' / 'file.txt'

# 路径信息
p.name              # 'file.txt'
p.stem              # 'file'
p.suffix            # '.txt'
p.parent            # 'folder'
p.exists()          # 是否存在
p.is_file()         # 是否是文件
p.is_dir()          # 是否是目录

# 操作
p.mkdir(parents=True, exist_ok=True)  # 创建目录
p.touch()           # 创建文件
p.rename('new.txt') # 重命名
p.unlink()          # 删除文件

# 遍历
for item in p.iterdir():
    print(item)

# 匹配
list(p.glob('*.txt'))
list(p.rglob('*.py'))  # 递归匹配
```

### json - JSON处理
```python
import json

# 序列化
data = {'name': 'Alice', 'age': 25}
json_str = json.dumps(data)
json_str = json.dumps(data, indent=2, ensure_ascii=False)

# 反序列化
data = json.loads(json_str)

# 文件操作
with open('data.json', 'w') as f:
    json.dump(data, f, indent=2)

with open('data.json', 'r') as f:
    data = json.load(f)
```

### re - 正则表达式
```python
import re

# 匹配
re.match(pattern, string)      # 从开头匹配
re.search(pattern, string)     # 查找第一个
re.findall(pattern, string)    # 查找所有
re.finditer(pattern, string)   # 返回迭代器

# 替换
re.sub(pattern, repl, string)

# 分割
re.split(pattern, string)

# 示例
pattern = r'\d+'
text = "我有123个苹果和456个橙子"
numbers = re.findall(pattern, text)  # ['123', '456']

# 编译正则（提高性能）
pattern = re.compile(r'\d+')
pattern.findall(text)
```

### os - 操作系统接口
```python
import os

# 文件系统
os.getcwd()             # 当前目录
os.chdir(path)          # 切换目录
os.listdir(path)        # 列出目录内容
os.mkdir(path)          # 创建目录
os.makedirs(path)       # 递归创建
os.remove(path)         # 删除文件
os.rmdir(path)          # 删除目录
os.rename(old, new)     # 重命名

# 路径操作
os.path.join('a', 'b', 'c')      # 连接路径
os.path.exists(path)             # 是否存在
os.path.isfile(path)             # 是否是文件
os.path.isdir(path)              # 是否是目录
os.path.basename(path)           # 文件名
os.path.dirname(path)            # 目录名
os.path.split(path)              # 分割
os.path.splitext(path)           # 分割扩展名

# 环境变量
os.environ['PATH']
os.getenv('PATH')
```

### sys - 系统相关
```python
import sys

sys.argv            # 命令行参数
sys.exit(code)      # 退出程序
sys.version         # Python版本
sys.platform        # 平台信息
sys.path            # 模块搜索路径
sys.stdin           # 标准输入
sys.stdout          # 标准输出
sys.stderr          # 标准错误
```

### random - 随机数
```python
import random

random.random()             # [0, 1) 浮点数
random.uniform(1, 10)       # [1, 10] 浮点数
random.randint(1, 10)       # [1, 10] 整数
random.choice([1, 2, 3])    # 随机选择
random.choices([1,2,3], k=2)  # 可重复选择
random.sample([1,2,3], k=2)   # 不重复选择
random.shuffle(lst)         # 打乱列表
```

### math - 数学函数
```python
import math

math.pi             # π
math.e              # 自然常数
math.ceil(3.2)      # 向上取整: 4
math.floor(3.8)     # 向下取整: 3
math.sqrt(16)       # 平方根: 4.0
math.pow(2, 3)      # 幂运算: 8.0
math.log(8, 2)      # 对数: 3.0
math.sin(math.pi/2) # 正弦: 1.0
math.cos(0)         # 余弦: 1.0
math.factorial(5)   # 阶乘: 120
```

---

## 十八、实用技巧

### 列表技巧
```python
# 扁平化嵌套列表
nested = [[1, 2], [3, 4], [5]]
flat = [item for sublist in nested for item in sublist]

# 去重保持顺序
def unique(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

# 分块
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# 转置矩阵
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = list(zip(*matrix))
```

### 字典技巧
```python
# 合并字典
d1 = {'a': 1, 'b': 2}
d2 = {'b': 3, 'c': 4}
merged = {**d1, **d2}       # Python 3.5+
merged = d1 | d2            # Python 3.9+

# 反转字典
d = {'a': 1, 'b': 2}
reversed_d = {v: k for k, v in d.items()}

# 字典排序
sorted_items = sorted(d.items(), key=lambda x: x[1])

# 默认值字典
from collections import defaultdict
d = defaultdict(int)
d['key'] += 1
```

### 字符串技巧
```python
# 多行字符串对齐
text = """
    Line 1
    Line 2
    Line 3
""".strip()

# 移除多余空格
" ".join(text.split())

# 检查子串
'hello' in text
text.startswith('prefix')
text.endswith('suffix')

# 填充
text.zfill(10)      # 左侧填充0
text.ljust(10)      # 右侧填充空格
text.center(10)     # 居中
```

### 性能优化
```python
# 使用生成器节省内存
sum(x**2 for x in range(1000000))  # 而非 sum([x**2 for x in range(1000000)])

# 使用集合提高查找速度
large_list = list(range(1000000))
large_set = set(large_list)
# 'x in large_set' 比 'x in large_list' 快得多

# 使用局部变量
def func():
    local_func = global_func  # 局部变量访问更快
    for i in range(1000):
        local_func(i)

# 字符串拼接
# 慢: s = ""; for x in lst: s += x
# 快: s = "".join(lst)
```

### 调试技巧
```python
# 打印变量名和值
def debug_print(var):
    import inspect
    frame = inspect.currentframe()
    name = [k for k, v in frame.f_back.f_locals.items() if v is var][0]
    print(f"{name} = {var}")

# 使用pprint美化输出
from pprint import pprint
pprint(complex_data)

# 使用断言
assert condition, "错误信息"

# 使用logging
import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("调试信息")
logging.info("普通信息")
logging.warning("警告")
logging.error("错误")
```

---

## 十九、最佳实践

### 代码风格（PEP 8）
```python
# 命名规范
variable_name       # 变量：小写+下划线
CONSTANT_NAME       # 常量：大写+下划线
function_name()     # 函数：小写+下划线
ClassName           # 类：驼峰命名
_private_var        # 私有：前导下划线

# 缩进：4个空格
if condition:
    do_something()

# 空行
# 类定义前后2个空行
# 方法定义前后1个空行

# 行长度：最多79字符
# 长语句可以用括号换行
result = (long_variable_name +
          another_long_name +
          yet_another_name)
```

### 异常处理
```python
# 具体异常优先
try:
    risky_operation()
except ValueError:
    handle_value_error()
except TypeError:
    handle_type_error()
except Exception as e:
    handle_general_error(e)
finally:
    cleanup()

# 不要捕获所有异常
# 不推荐: except:
# 推荐: except Exception:

# 自定义异常
class CustomError(Exception):
    """自定义异常说明"""
    pass
```

### 性能考虑
```python
# 使用列表推导式代替循环
# 慢
result = []
for x in range(100):
    result.append(x**2)

# 快
result = [x**2 for x in range(100)]

# 避免在循环中重复计算
# 慢
for i in range(len(lst)):
    if i < len(lst) - 1:
        process(lst[i])

# 快
length = len(lst)
for i in range(length):
    if i < length - 1:
        process(lst[i])
```

### 资源管理
```python
# 使用 with 语句
with open('file.txt') as f:
    content = f.read()

# 多个资源
with open('in.txt') as f_in, open('out.txt', 'w') as f_out:
    f_out.write(f_in.read())
```

---

## 二十、常见陷阱

### 可变默认参数
```python
# 错误
def append_to(element, lst=[]):
    lst.append(element)
    return lst

# 正确
def append_to(element, lst=None):
    if lst is None:
        lst = []
    lst.append(element)
    return lst
```

### 闭包陷阱
```python
# 错误
funcs = [lambda: i for i in range(5)]
[f() for f in funcs]  # 全是4

# 正确
funcs = [lambda i=i: i for i in range(5)]
[f() for f in funcs]  # [0, 1, 2, 3, 4]
```

### 浅拷贝vs深拷贝
```python
import copy

# 浅拷贝
lst1 = [[1, 2], [3, 4]]
lst2 = lst1.copy()
lst2[0][0] = 999  # lst1也会改变

# 深拷贝
lst3 = copy.deepcopy(lst1)
lst3[0][0] = 999  # lst1不变
```

### is vs ==
```python
# is 比较身份
# == 比较值
a = [1, 2, 3]
b = [1, 2, 3]
a == b  # True
a is b  # False

# 小整数和字符串有缓存
x = 256
y = 256
x is y  # True（小整数缓存）

x = 257
y = 257
x is y  # False（超出缓存范围）
```

---

## 总结

这份笔记涵盖了Python的核心语法和常用特性：

1. **基础部分**：变量、数据类型、运算符、控制流
2. **数据结构**：列表、元组、字典、集合及其详细操作
3. **函数与面向对象**：函数定义、类、继承、特殊方法
4. **并发编程**：多线程、多进程、协程的使用和对比
5. **高级特性**：推导式、生成器、装饰器、上下文管理器
6. **标准库**：常用模块的使用方法
7. **最佳实践**：代码风格、性能优化、常见陷阱

**学习建议**：
- 先掌握基础语法和数据结构
- 多写代码实践，理解概念
- 逐步学习高级特性
- 阅读优秀的开源代码
- 参考官方文档：https://docs.python.org

祝你学习愉快！🐍

---

## 二十一、文件和目录操作进阶

### 文件读写模式详解
```python
# 文本模式
'r'   # 只读（默认）
'w'   # 写入（覆盖）
'a'   # 追加
'x'   # 独占创建（文件存在则失败）
'r+'  # 读写
'w+'  # 读写（覆盖）
'a+'  # 读写（追加）

# 二进制模式（加 'b'）
'rb'  # 二进制读
'wb'  # 二进制写
'ab'  # 二进制追加

# 示例
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()

with open('file.bin', 'rb') as f:
    binary_data = f.read()
```

### 文件指针操作
```python
with open('file.txt', 'r') as f:
    f.seek(0)           # 移到开头
    f.seek(10)          # 移到第10字节
    f.seek(0, 2)        # 移到末尾（0相对于末尾）
    pos = f.tell()      # 获取当前位置
    f.read(100)         # 读取100字节
```

### CSV文件操作
```python
import csv

# 读取CSV
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)      # 读取表头
    for row in reader:
        print(row)

# 使用DictReader（推荐）
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['column_name'])

# 写入CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerows([
        ['Alice', 25, 'Beijing'],
        ['Bob', 30, 'Shanghai']
    ])

# 使用DictWriter
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['Name', 'Age', 'City']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Name': 'Alice', 'Age': 25, 'City': 'Beijing'})
```

### 压缩文件操作
```python
import zipfile
import tarfile
import gzip

# ZIP文件
with zipfile.ZipFile('archive.zip', 'w') as zipf:
    zipf.write('file1.txt')
    zipf.write('file2.txt')

with zipfile.ZipFile('archive.zip', 'r') as zipf:
    zipf.extractall('output_dir')
    names = zipf.namelist()

# TAR文件
with tarfile.open('archive.tar.gz', 'w:gz') as tar:
    tar.add('folder/')

with tarfile.open('archive.tar.gz', 'r:gz') as tar:
    tar.extractall('output_dir')

# GZIP单文件
with gzip.open('file.txt.gz', 'wt') as f:
    f.write('content')

with gzip.open('file.txt.gz', 'rt') as f:
    content = f.read()
```

### 临时文件
```python
import tempfile

# 临时文件
with tempfile.TemporaryFile(mode='w+t') as f:
    f.write('temp data')
    f.seek(0)
    data = f.read()
# 自动删除

# 命名临时文件
with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write('data')
    temp_name = f.name

# 临时目录
with tempfile.TemporaryDirectory() as tmpdir:
    print(f'临时目录: {tmpdir}')
    # 使用临时目录
# 自动删除
```

---

## 二十二、网络编程

### HTTP请求（requests库）
```python
import requests

# GET请求
response = requests.get('https://api.example.com/data')
print(response.status_code)
print(response.text)
print(response.json())
print(response.headers)

# 带参数
params = {'key': 'value', 'page': 1}
response = requests.get('https://api.example.com/data', params=params)

# POST请求
data = {'username': 'admin', 'password': '123'}
response = requests.post('https://api.example.com/login', data=data)

# JSON数据
json_data = {'key': 'value'}
response = requests.post('https://api.example.com/api', json=json_data)

# 上传文件
files = {'file': open('report.pdf', 'rb')}
response = requests.post('https://api.example.com/upload', files=files)

# 自定义请求头
headers = {'User-Agent': 'MyApp/1.0', 'Authorization': 'Bearer token'}
response = requests.get('https://api.example.com/data', headers=headers)

# 会话（保持Cookie）
session = requests.Session()
session.get('https://example.com/login')
session.post('https://example.com/action')

# 超时和重试
response = requests.get('https://api.example.com', timeout=5)

# 处理异常
try:
    response = requests.get('https://api.example.com')
    response.raise_for_status()  # 4xx或5xx会抛出异常
except requests.exceptions.HTTPError as e:
    print(f"HTTP错误: {e}")
except requests.exceptions.ConnectionError:
    print("连接错误")
except requests.exceptions.Timeout:
    print("超时")
except requests.exceptions.RequestException as e:
    print(f"请求错误: {e}")
```

### Socket编程
```python
import socket

# TCP服务器
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 8888))
server.listen(5)

while True:
    client, addr = server.accept()
    print(f"连接来自: {addr}")
    data = client.recv(1024)
    client.send(b"Hello from server")
    client.close()

# TCP客户端
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 8888))
client.send(b"Hello from client")
response = client.recv(1024)
client.close()

# UDP通信
# 服务器
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('localhost', 9999))
data, addr = sock.recvfrom(1024)
sock.sendto(b"Response", addr)

# 客户端
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b"Message", ('localhost', 9999))
data, addr = sock.recvfrom(1024)
```

### Web框架基础（Flask示例）
```python
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# 路由
@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/user/<username>')
def show_user(username):
    return f'User: {username}'

# GET和POST
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        return f'Logged in as {username}'
    return render_template('login.html')

# JSON API
@app.route('/api/data')
def get_data():
    return jsonify({'key': 'value', 'items': [1, 2, 3]})

# 查询参数
@app.route('/search')
def search():
    query = request.args.get('q', '')
    return f'Searching for: {query}'

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## 二十三、数据处理

### 数据序列化
```python
import pickle
import json
import yaml  # 需要安装: pip install pyyaml

# Pickle（Python特有）
data = {'name': 'Alice', 'scores': [90, 85, 88]}

# 保存
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

# 读取
with open('data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# JSON
with open('data.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

with open('data.json', 'r') as f:
    loaded_data = json.load(f)

# YAML
with open('config.yaml', 'w') as f:
    yaml.dump(data, f)

with open('config.yaml', 'r') as f:
    loaded_data = yaml.safe_load(f)
```

### 配置文件处理
```python
import configparser

# INI格式
config = configparser.ConfigParser()

# 写入
config['DEFAULT'] = {'ServerAliveInterval': '45'}
config['database'] = {
    'host': 'localhost',
    'port': '3306',
    'user': 'admin'
}

with open('config.ini', 'w') as f:
    config.write(f)

# 读取
config.read('config.ini')
host = config['database']['host']
port = config.getint('database', 'port')

# 环境变量
import os
from dotenv import load_dotenv  # pip install python-dotenv

# .env文件内容：
# DATABASE_URL=postgresql://user:pass@localhost/db
# SECRET_KEY=mysecret

load_dotenv()
db_url = os.getenv('DATABASE_URL')
secret = os.getenv('SECRET_KEY', 'default_secret')
```

### 命令行参数解析
```python
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='处理数据的脚本')

# 添加参数
parser.add_argument('input', help='输入文件')
parser.add_argument('-o', '--output', help='输出文件', default='output.txt')
parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
parser.add_argument('-n', '--number', type=int, default=10, help='数量')
parser.add_argument('--format', choices=['json', 'csv', 'xml'], default='json')

# 解析
args = parser.parse_args()

print(f"输入: {args.input}")
print(f"输出: {args.output}")
if args.verbose:
    print("详细模式")
print(f"数量: {args.number}")

# 使用: python script.py input.txt -o output.txt -v -n 20 --format csv
```

---

## 二十四、数据库操作

### SQLite（内置）
SQLite 是一个轻量级的、无服务器的、自包含的 SQL 数据库引擎，非常适合小型应用、原型开发和数据分析。

```python
import sqlite3

# 1. 连接数据库（如果文件不存在，会自动创建）
conn = sqlite3.connect('example.db')
# 创建一个游标对象，用于执行SQL语句
cursor = conn.cursor()

# 2. 创建表
# 使用 """ 多行字符串编写 SQL
# IF NOT EXISTS 确保表只在不存在时创建，避免重复执行报错
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        email TEXT UNIQUE
    )
''')

# 3. 插入数据
# 使用 ? 作为占位符进行参数化查询，可以防止SQL注入，这是安全最佳实践
cursor.execute("INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
               ('Alice', 25, 'alice@example.com'))

# 批量插入
users_to_insert = [
    ('Bob', 30, 'bob@example.com'),
    ('Charlie', 35, 'charlie@example.com')
]
cursor.executemany("INSERT INTO users (name, age, email) VALUES (?, ?, ?)", users_to_insert)

# 4. 提交事务
# 对数据库的所有修改都需要提交后才会生效
conn.commit()

# 5. 查询数据
# 查询所有用户
cursor.execute("SELECT * FROM users")
all_users = cursor.fetchall()  # 获取所有结果行
for user in all_users:
    print(user)  # 输出: (1, 'Alice', 25, 'alice@example.com'), ...

# 条件查询
cursor.execute("SELECT name, email FROM users WHERE age > ?", (28,))
some_users = cursor.fetchall()
print(some_users) # 输出: [('Bob', 'bob@example.com'), ('Charlie', 'charlie@example.com')]

# 查询单条记录
cursor.execute("SELECT * FROM users WHERE name = ?", ('Alice',))
alice = cursor.fetchone() # 获取第一条结果
print(alice)

# 6. 更新数据
cursor.execute("UPDATE users SET age = ? WHERE name = ?", (26, 'Alice'))
conn.commit()

# 7. 删除数据
cursor.execute("DELETE FROM users WHERE name = ?", ('Charlie',))
conn.commit()

# 8. 关闭连接
# 操作完成后，务必关闭游标和连接
cursor.close()
conn.close()

# 推荐使用 with 语句自动管理连接和事务
try:
    with sqlite3.connect('example.db') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
                       ('David', 40, 'david@example.com'))
        # with 语句块结束时会自动提交事务，如果发生异常则会自动回滚
except sqlite3.Error as e:
    print(f"数据库错误: {e}")

```

#### ORM (对象关系映射) - SQLAlchemy 示例
ORM 允许你使用 Python 对象来操作数据库，而无需编写原生 SQL 语句，使代码更具可读性和可维护性。SQLAlchemy 是 Python 中最流行的 ORM 框架。

首先，你需要安装它：`pip install sqlalchemy`

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. 设置数据库连接
# 创建一个引擎，连接到我们的 SQLite 数据库
engine = create_engine('sqlite:///example.db')

# 2. 定义数据模型 (ORM 类)
# 创建一个基类，我们的 ORM 模型将继承它
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'  # 关联到数据库中的 'users' 表

    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)
    email = Column(String, unique=True)

    def __repr__(self):
        return f"<User(name='{self.name}', age={self.age})>"

# 3. 创建表结构 (如果不存在)
# 这会检查数据库，并创建所有继承自 Base 的模型对应的表
Base.metadata.create_all(engine)

# 4. 创建会话 (Session)
# Session 是与数据库交互的主要入口
Session = sessionmaker(bind=engine)
session = Session()

# 5. 插入数据 (创建对象)
new_user_eve = User(name='Eve', age=28, email='eve@example.com')
session.add(new_user_eve)

# 批量添加
session.add_all([
    User(name='Frank', age=45, email='frank@example.com'),
    User(name='Grace', age=32, email='grace@example.com')
])

session.commit() # 提交事务

# 6. 查询数据 (查询对象)
# 查询所有用户
all_users = session.query(User).all()
print(all_users)

# 条件查询
users_over_30 = session.query(User).filter(User.age > 30).all()
print(users_over_30)

# 查询第一个匹配项
frank = session.query(User).filter_by(name='Frank').first()
print(frank)

# 7. 更新数据 (修改对象属性)
if frank:
    frank.age = 46
    session.commit()

# 8. 删除数据
grace = session.query(User).filter_by(name='Grace').first()
if grace:
    session.delete(grace)
    session.commit()

# 9. 关闭会话
session.close()
```