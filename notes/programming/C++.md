# C++ 完整语法笔记

## 目录
1. [基础语法](#一基础语法)
2. [数据类型](#二数据类型)
3. [控制流](#三控制流)
4. [函数](#四函数)
5. [数组与指针](#五数组与指针)
6. [引用](#六引用)
7. [面向对象](#七面向对象)
8. [内存管理](#八内存管理)
9. [STL容器](#九stl容器)
10. [STL算法](#十stl算法)
11. [模板](#十一模板)
12. [现代C++特性](#十二现代c特性)

---

## 一、基础语法

### 程序结构
```cpp
#include <iostream>  // 预处理指令

using namespace std;  // 命名空间

// 主函数
int main() {
    cout << "Hello, World!" << endl;
    return 0;
}

// 不使用 using namespace
int main() {
    std::cout << "Hello!" << std::endl;
    return 0;
}
```

### 变量声明
```cpp
// 基本类型
int x = 10;           // 整数
double y = 3.14;      // 双精度浮点
float z = 3.14f;      // 单精度浮点
char c = 'A';         // 字符
bool flag = true;     // 布尔值

// 类型推导（C++11）
auto a = 10;          // int
auto b = 3.14;        // double
auto c = "hello";     // const char*

// 常量
const int MAX = 100;
constexpr int SIZE = 10;  // 编译期常量（C++11）

// 类型别名
typedef unsigned int uint;
using uint = unsigned int;  // C++11 推荐

// 变量初始化方式
int x = 10;           // 传统方式
int y(20);            // 构造函数方式
int z{30};            // 统一初始化（C++11）
int w = {40};         // 统一初始化

// 多变量声明
int a = 1, b = 2, c = 3;
int* p1, p2;          // p1是指针，p2不是！
int *p1, *p2;         // 都是指针
```

### 输入输出
```cpp
#include <iostream>
using namespace std;

// 输出
cout << "Hello" << endl;
cout << "x = " << x << ", y = " << y << endl;

// 输入
int num;
cin >> num;

string name;
cin >> name;          // 读取一个单词
getline(cin, name);   // 读取一行

// 格式化输出
#include <iomanip>
cout << fixed << setprecision(2) << 3.14159;  // 3.14
cout << setw(10) << 123;  // 右对齐，宽度10
```

### 注释
```cpp
// 单行注释

/*
多行注释
可以跨越多行
*/

/**
 * 文档注释
 * @param x 参数说明
 * @return 返回值说明
 */
```

### 运算符
```cpp
// 算术运算符
+    // 加
-    // 减
*    // 乘
/    // 除
%    // 取余（仅整数）
++   // 自增
--   // 自减

// 比较运算符
==   // 等于
!=   // 不等于
>    // 大于
<    // 小于
>=   // 大于等于
<=   // 小于等于

// 逻辑运算符
&&   // 与
||   // 或
!    // 非

// 位运算符
&    // 按位与
|    // 按位或
^    // 按位异或
~    // 按位取反
<<   // 左移
>>   // 右移

// 赋值运算符
=    // 赋值
+=   // 加等于
-=   // 减等于
*=   // 乘等于
/=   // 除等于
%=   // 取余等于
&=   // 按位与等于
|=   // 按位或等于
^=   // 按位异或等于
<<=  // 左移等于
>>=  // 右移等于

// 三元运算符
condition ? value1 : value2

// sizeof运算符
sizeof(int);         // 4（通常）
sizeof(variable);
```

---

## 二、数据类型

### 基本数据类型
```cpp
// 整数类型
char          // 1字节  -128 到 127
short         // 2字节  -32768 到 32767
int           // 4字节  -2^31 到 2^31-1
long          // 4或8字节
long long     // 8字节  （C++11）

// 无符号整数
unsigned char
unsigned short
unsigned int
unsigned long
unsigned long long

// 浮点类型
float         // 4字节  约7位有效数字
double        // 8字节  约15位有效数字
long double   // 10/12/16字节

// 布尔类型
bool          // true 或 false

// 字符类型
char          // 单字节字符
wchar_t       // 宽字符
char16_t      // UTF-16字符（C++11）
char32_t      // UTF-32字符（C++11）

// void类型
void          // 无类型

// 类型大小
#include <limits>
cout << "int最大值: " << INT_MAX << endl;
cout << "int最小值: " << INT_MIN << endl;
cout << "double精度: " << DBL_DIG << endl;
```

### 字符串
```cpp
// C风格字符串
char str1[] = "Hello";
char str2[10] = "World";
const char* str3 = "Hello";

// C++字符串（推荐）
#include <string>
string s1 = "Hello";
string s2("World");
string s3(5, 'A');    // "AAAAA"

// 字符串操作
s1.length()           // 长度
s1.size()             // 长度
s1.empty()            // 是否为空
s1.clear()            // 清空

s1[0]                 // 访问字符
s1.at(0)              // 安全访问（会检查边界）

s1 + s2               // 连接
s1.append(s2)         // 追加
s1 += s2              // 追加

s1.substr(0, 5)       // 子串
s1.find("lo")         // 查找，返回位置
s1.rfind("lo")        // 从后查找
s1.replace(0, 2, "Hi") // 替换

s1.compare(s2)        // 比较
s1 == s2              // 相等比较
s1 < s2               // 字典序比较

// 转换
to_string(123)        // 数字转字符串（C++11）
stoi("123")           // 字符串转int（C++11）
stod("3.14")          // 字符串转double（C++11）
stol("123")           // 字符串转long
stoll("123")          // 字符串转long long

// C风格字符串操作
#include <cstring>
strlen(str)           // 长度
strcpy(dest, src)     // 复制
strcat(dest, src)     // 连接
strcmp(str1, str2)    // 比较
```

### 枚举
```cpp
// 传统枚举
enum Color {
    RED,      // 0
    GREEN,    // 1
    BLUE      // 2
};
Color c = RED;

enum Status {
    OK = 200,
    NOT_FOUND = 404,
    ERROR = 500
};

// 强类型枚举（C++11）
enum class TrafficLight {
    Red,
    Yellow,
    Green
};

TrafficLight light = TrafficLight::Red;

// 指定底层类型
enum class Byte : unsigned char {
    B0, B1, B2
};
```

### 结构体
```cpp
// 定义结构体
struct Point {
    int x;
    int y;
};

// 使用
Point p1;
p1.x = 10;
p1.y = 20;

Point p2 = {30, 40};      // 聚合初始化
Point p3{50, 60};         // C++11 统一初始化

// 结构体指针
Point* ptr = &p1;
ptr->x = 100;
(*ptr).y = 200;

// 结构体可以有成员函数
struct Rectangle {
    int width;
    int height;
    
    int area() {
        return width * height;
    }
};
```

### 联合体
```cpp
union Data {
    int i;
    float f;
    char str[20];
};

Data d;
d.i = 10;
d.f = 3.14;  // 覆盖之前的值
```

---

## 三、控制流

### 条件语句
```cpp
// if-else
if (condition) {
    // 代码块
} else if (another_condition) {
    // 代码块
} else {
    // 代码块
}

// 三元运算符
int max = (a > b) ? a : b;

// switch
switch (value) {
    case 1:
        // 代码块
        break;
    case 2:
    case 3:
        // 多个case共享
        break;
    default:
        // 默认
        break;
}

// C++17: if with initializer
if (auto result = getValue(); result > 0) {
    // result 只在此作用域有效
}
```

### 循环
```cpp
// for循环
for (int i = 0; i < 10; i++) {
    cout << i << endl;
}

// 范围for循环（C++11）
vector<int> nums = {1, 2, 3, 4, 5};
for (int n : nums) {
    cout << n << endl;
}

for (int& n : nums) {  // 引用，可修改
    n *= 2;
}

for (const auto& n : nums) {  // 常量引用，不拷贝
    cout << n << endl;
}

// while循环
while (condition) {
    // 代码块
}

// do-while循环
do {
    // 至少执行一次
} while (condition);

// 控制语句
break;      // 跳出循环
continue;   // 跳过本次迭代

// goto（不推荐）
goto label;
// ...
label:
    // 代码
```

---

## 四、函数

### 函数定义
```cpp
// 函数声明（原型）
int add(int a, int b);

// 函数定义
int add(int a, int b) {
    return a + b;
}

// 无返回值
void printMessage(string msg) {
    cout << msg << endl;
}

// 默认参数
void greet(string name = "Guest") {
    cout << "Hello, " << name << endl;
}

// 多个默认参数（从右到左）
void func(int a, int b = 10, int c = 20);

// 内联函数
inline int square(int x) {
    return x * x;
}
```

### 函数重载
```cpp
// 同名函数，参数不同
int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

int add(int a, int b, int c) {
    return a + b + c;
}

// 调用
add(1, 2);        // 调用 int 版本
add(1.0, 2.0);    // 调用 double 版本
add(1, 2, 3);     // 调用三参数版本
```

### 函数指针
```cpp
// 函数指针声明
int (*funcPtr)(int, int);

// 赋值
funcPtr = add;

// 调用
int result = funcPtr(3, 4);
int result = (*funcPtr)(3, 4);  // 等价

// 函数指针数组
int (*operations[4])(int, int) = {add, sub, mul, div};

// 回调函数
void processArray(int arr[], int size, int (*func)(int)) {
    for (int i = 0; i < size; i++) {
        arr[i] = func(arr[i]);
    }
}
```

### Lambda表达式（C++11）
```cpp
// 基本语法: [捕获](参数) -> 返回类型 { 函数体 }

// 简单lambda
auto add = [](int a, int b) { return a + b; };
int result = add(3, 4);

// 自动推导返回类型
auto multiply = [](int a, int b) { return a * b; };

// 显式指定返回类型
auto divide = [](int a, int b) -> double {
    return static_cast<double>(a) / b;
};

// 捕获外部变量
int x = 10;
auto addX = [x](int n) { return x + n; };  // 值捕获

int y = 20;
auto addY = [&y](int n) { return y + n; };  // 引用捕获
y = 30;  // 会影响lambda

// 捕获方式
[]        // 不捕获
[=]       // 值捕获所有
[&]       // 引用捕获所有
[x]       // 值捕获x
[&x]      // 引用捕获x
[x, &y]   // x值捕获，y引用捕获
[=, &y]   // 默认值捕获，y引用捕获

// mutable（修改值捕获的变量）
int count = 0;
auto increment = [count]() mutable { return ++count; };

// 泛型lambda（C++14）
auto add = [](auto a, auto b) { return a + b; };
```

### 递归
```cpp
// 阶乘
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// 斐波那契
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// 尾递归
int factorial_tail(int n, int acc = 1) {
    if (n <= 1) return acc;
    return factorial_tail(n - 1, n * acc);
}
```

---

## 五、数组与指针

### 数组
```cpp
// 一维数组
int arr[5];                    // 声明
int arr[5] = {1, 2, 3, 4, 5}; // 初始化
int arr[] = {1, 2, 3};         // 自动推导大小
int arr[5] = {1, 2};           // 部分初始化，其余为0

// 访问
arr[0] = 10;
int x = arr[2];

// 遍历
for (int i = 0; i < 5; i++) {
    cout << arr[i] << endl;
}

// 数组大小
int size = sizeof(arr) / sizeof(arr[0]);

// 多维数组
int matrix[3][4];
int matrix[3][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
};

// 访问
matrix[0][0] = 1;
int value = matrix[1][2];

// 数组作为函数参数
void printArray(int arr[], int size) {
    // arr实际上是指针
}

// C++11: std::array（推荐）
#include <array>
array<int, 5> arr = {1, 2, 3, 4, 5};
arr[0] = 10;
arr.size();
arr.at(2);  // 边界检查
```

### 指针
```cpp
// 指针声明
int* ptr;              // 指向int的指针
int *p1, *p2;         // 两个指针
int* p3, p4;          // p3是指针，p4不是

// 取地址
int x = 10;
int* ptr = &x;

// 解引用
*ptr = 20;            // 修改x的值
int y = *ptr;         // 读取x的值

// 空指针
int* ptr = nullptr;   // C++11
int* ptr = NULL;      // 传统方式
int* ptr = 0;         // 传统方式

// 指针运算
int arr[] = {1, 2, 3, 4, 5};
int* ptr = arr;       // 数组名是指向首元素的指针

ptr++;                // 指向下一个元素
ptr--;                // 指向上一个元素
ptr + 2;              // 向后移动2个元素
ptr - 1;              // 向前移动1个元素

// 指针比较
ptr1 == ptr2
ptr1 < ptr2

// 指针和数组
arr[i] == *(arr + i)
ptr[i] == *(ptr + i)

// 指向指针的指针
int x = 10;
int* p = &x;
int** pp = &p;
**pp = 20;            // 修改x

// const指针
const int* ptr;       // 指向常量的指针（不能通过ptr修改值）
int* const ptr;       // 常量指针（不能修改指针本身）
const int* const ptr; // 两者都是常量

// 函数指针
int (*funcPtr)(int, int);
```

### 动态内存分配
```cpp
// new和delete
int* ptr = new int;          // 分配单个int
*ptr = 10;
delete ptr;                  // 释放

int* arr = new int[10];      // 分配数组
arr[0] = 1;
delete[] arr;                // 释放数组

// 初始化
int* ptr = new int(10);      // 初始化为10
int* arr = new int[5]{1, 2, 3, 4, 5};  // C++11

// 动态二维数组
int** matrix = new int*[rows];
for (int i = 0; i < rows; i++) {
    matrix[i] = new int[cols];
}

// 释放
for (int i = 0; i < rows; i++) {
    delete[] matrix[i];
}
delete[] matrix;

// 检查分配失败
int* ptr = new (nothrow) int[1000000000];
if (ptr == nullptr) {
    cout << "分配失败" << endl;
}
```

---

## 六、引用

### 引用基础
```cpp
// 引用声明（必须初始化）
int x = 10;
int& ref = x;         // ref是x的别名

ref = 20;             // 修改x
cout << x;            // 输出20

// 不能重新绑定
int y = 30;
ref = y;              // 这是赋值，不是重新绑定

// 常量引用
const int& cref = x;
// cref = 40;         // 错误：不能修改

// 临时对象的常量引用
const int& ref = 10;  // 可以绑定到临时对象
```

### 引用作为函数参数
```cpp
// 值传递
void func1(int x) {
    x = 100;          // 不影响原变量
}

// 引用传递
void func2(int& x) {
    x = 100;          // 修改原变量
}

// 常量引用（避免拷贝，不修改）
void func3(const string& s) {
    cout << s << endl;
    // s += "!";      // 错误：不能修改
}

// 使用
int a = 10;
func1(a);             // a还是10
func2(a);             // a变成100

string str = "Hello";
func3(str);           // 不拷贝，高效
```

### 引用作为返回值
```cpp
// 返回引用
int& getElement(vector<int>& vec, int index) {
    return vec[index];
}

vector<int> nums = {1, 2, 3};
getElement(nums, 0) = 100;  // 可以修改

// 不要返回局部变量的引用
int& bad() {
    int x = 10;
    return x;         // ❌ 危险：返回局部变量引用
}

// 链式调用
class Counter {
    int count;
public:
    Counter& increment() {
        count++;
        return *this;
    }
};

Counter c;
c.increment().increment().increment();
```

### 右值引用（C++11）
```cpp
// 左值和右值
int x = 10;           // x是左值
int y = x + 1;        // x+1是右值

// 右值引用
int&& rref = 10;      // 绑定到右值
int&& rref = x + 1;   // 绑定到临时对象

// 移动语义
class String {
    char* data;
public:
    // 移动构造函数
    String(String&& other) noexcept {
        data = other.data;
        other.data = nullptr;
    }
    
    // 移动赋值运算符
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
};

// std::move
String s1 = "Hello";
String s2 = std::move(s1);  // s1的资源转移到s2
```

---

## 七、面向对象

### 类定义
```cpp
class Person {
private:
    string name;
    int age;
    
public:
    // 构造函数
    Person() : name(""), age(0) {}
    
    Person(string n, int a) : name(n), age(a) {}
    
    // 成员函数
    void setName(string n) {
        name = n;
    }
    
    string getName() const {  // const成员函数
        return name;
    }
    
    void printInfo() {
        cout << name << ", " << age << endl;
    }
    
    // 析构函数
    ~Person() {
        // 清理资源
    }
};

// 使用
Person p1;                      // 默认构造
Person p2("Alice", 25);         // 参数构造
Person p3 = Person("Bob", 30);  // 显式构造

p2.setName("Alice Wang");
cout << p2.getName() << endl;
```

### 访问控制
```cpp
class MyClass {
private:
    int privateVar;      // 只能类内访问
    
protected:
    int protectedVar;    // 类内和派生类可访问
    
public:
    int publicVar;       // 任何地方可访问
    
    void publicFunc();
};
```

### 构造函数和析构函数
```cpp
class Rectangle {
private:
    int width, height;
    
public:
    // 默认构造函数
    Rectangle() : width(0), height(0) {
        cout << "默认构造" << endl;
    }
    
    // 参数构造函数
    Rectangle(int w, int h) : width(w), height(h) {
        cout << "参数构造" << endl;
    }
    
    // 拷贝构造函数
    Rectangle(const Rectangle& other)
        : width(other.width), height(other.height) {
        cout << "拷贝构造" << endl;
    }
    
    // 移动构造函数（C++11）
    Rectangle(Rectangle&& other) noexcept
        : width(other.width), height(other.height) {
        cout << "移动构造" << endl;
        other.width = 0;
        other.height = 0;
    }
    
    // 析构函数
    ~Rectangle() {
        cout << "析构" << endl;
    }
    
    // 委托构造函数（C++11）
    Rectangle(int size) : Rectangle(size, size) {
        // 委托给另一个构造函数
    }
};

// Rule of Three/Five
// 如果需要自定义以下任一个，通常都需要自定义：
// 1. 析构函数
// 2. 拷贝构造函数
// 3. 拷贝赋值运算符
// C++11新增：
// 4. 移动构造函数
// 5. 移动赋值运算符
```

### 运算符重载
```cpp
class Complex {
private:
    double real, imag;
    
public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}
    
    // 成员函数重载
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }
    
    // 前置++
    Complex& operator++() {
        ++real;
        return *this;
    }
    
    // 后置++
    Complex operator++(int) {
        Complex temp = *this;
        ++real;
        return temp;
    }
    
    // 赋值运算符
    Complex& operator=(const Complex& other) {
        if (this != &other) {
            real = other.real;
            imag = other.imag;
        }
        return *this;
    }
    
    // 比较运算符
    bool operator==(const Complex& other) const {
        return real == other.real && imag == other.imag;
    }
    
    // 下标运算符
    double& operator[](int index) {
        return index == 0 ? real : imag;
    }
    
    // 函数调用运算符
    double operator()() const {
        return sqrt(real * real + imag * imag);
    }
    
    // 类型转换运算符
    operator double() const {
        return real;
    }
    
    // 友元函数重载（非成员）
    friend ostream& operator<<(ostream& os, const Complex& c) {
        os << c.real << " + " << c.imag << "i";
        return os;
    }
    
    friend Complex operator*(const Complex& c1, const Complex& c2);
};

// 非成员运算符重载
Complex operator*(const Complex& c1, const Complex& c2) {
    return Complex(
        c1.real * c2.real - c1.imag * c2.imag,
        c1.real * c2.imag + c1.imag * c2.real
    );
}
```

### 继承
```cpp
// 基类
class Animal {
protected:
    string name;
    
public:
    Animal(string n) : name(n) {}
    
    void eat() {
        cout << name << " is eating" << endl;
    }
    
    virtual void makeSound() {  // 虚函数
        cout << "Some sound" << endl;
    }
    
    virtual ~Animal() {}  // 虚析构函数
};

// 派生类
class Dog : public Animal {
private:
    string breed;
    
public:
    Dog(string n, string b) : Animal(n), breed(b) {}
    
    // 重写虚函数
    void makeSound() override {  // C++11: override关键字
        cout << "Woof!" << endl;
    }
    
    void wagTail() {
        cout << name << " wags tail" << endl;
    }
};

// 多重继承
class Cat : public Animal {
public:
    Cat(string n) : Animal(n) {}
    void makeSound() override {
        cout << "Meow!" << endl;
    }
};

class Robot {
public:
    void charge() {
        cout << "Charging..." << endl;
    }
};

class RoboCat : public Cat, public Robot {
public:
    RoboCat(string n) : Cat(n) {}
};

// 使用
Dog dog("Buddy", "Golden Retriever");
dog.eat();
dog.makeSound();
dog.wagTail();

Animal* ptr = &dog;
ptr->makeSound();  // 多态：调用Dog的版本
```

### 多态
```cpp
// 虚函数实现多态
class Shape {
public:
    virtual double area() const = 0;  // 纯虚函数
    virtual void draw() const = 0;
    virtual ~Shape() {}
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(double r) : radius(r) {}
    
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    void draw() const override {
        cout << "Drawing circle" << endl;
    }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(double w