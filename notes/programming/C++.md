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
    Rectangle(double w, double h) : width(w), height(h) {}
    
    double area() const override {
        return width * height;
    }
    
    void draw() const override {
        cout << "Drawing rectangle" << endl;
    }
};

// 使用多态
void printArea(const Shape& shape) {
    cout << "Area: " << shape.area() << endl;
    shape.draw();
}

Circle c(5);
Rectangle r(4, 6);
printArea(c);  // 调用Circle的实现
printArea(r);  // 调用Rectangle的实现

// 动态绑定
Shape* shapes[2];
shapes[0] = new Circle(5);
shapes[1] = new Rectangle(4, 6);

for (int i = 0; i < 2; i++) {
    shapes[i]->draw();  // 运行时决定调用哪个版本
    delete shapes[i];
}
```

### 抽象类和接口
```cpp
// 抽象类（包含纯虚函数）
class AbstractClass {
public:
    virtual void pureVirtualFunc() = 0;  // 纯虚函数
    virtual void virtualFunc() {         // 普通虚函数
        cout << "Default implementation" << endl;
    }
    virtual ~AbstractClass() {}
};

// 不能实例化抽象类
// AbstractClass obj;  // 错误

// 接口（所有函数都是纯虚函数）
class IDrawable {
public:
    virtual void draw() = 0;
    virtual ~IDrawable() {}
};

class IPrintable {
public:
    virtual void print() = 0;
    virtual ~IPrintable() {}
};

// 实现多个接口
class Document : public IDrawable, public IPrintable {
public:
    void draw() override {
        cout << "Drawing document" << endl;
    }
    
    void print() override {
        cout << "Printing document" << endl;
    }
};
```

### 友元
```cpp
class Box {
private:
    double width;
    
    // 友元函数
    friend void printWidth(Box& b);
    
    // 友元类
    friend class BoxManager;
    
public:
    Box(double w) : width(w) {}
};

// 友元函数可以访问私有成员
void printWidth(Box& b) {
    cout << "Width: " << b.width << endl;
}

// 友元类可以访问所有私有成员
class BoxManager {
public:
    void setWidth(Box& b, double w) {
        b.width = w;
    }
    
    double getWidth(Box& b) {
        return b.width;
    }
};
```

### 静态成员
```cpp
class Counter {
private:
    static int count;  // 静态成员变量声明
    int id;
    
public:
    Counter() {
        id = ++count;
    }
    
    static int getCount() {  // 静态成员函数
        return count;
        // 不能访问非静态成员
        // return id;  // 错误
    }
    
    int getId() const {
        return id;
    }
};

// 静态成员变量定义（在类外）
int Counter::count = 0;

// 使用
Counter c1, c2, c3;
cout << Counter::getCount() << endl;  // 3
cout << c1.getId() << endl;           // 1
cout << c2.getId() << endl;           // 2
```

### 嵌套类
```cpp
class Outer {
private:
    int outerData;
    
    class Inner {  // 嵌套类
    private:
        int innerData;
        
    public:
        Inner(int data) : innerData(data) {}
        
        void display() {
            cout << "Inner: " << innerData << endl;
        }
    };
    
public:
    Outer(int data) : outerData(data) {}
    
    void createInner() {
        Inner inner(outerData);
        inner.display();
    }
};
```

---

## 八、内存管理

### 智能指针（C++11）
```cpp
#include <memory>

// unique_ptr（独占所有权）
unique_ptr<int> ptr1(new int(10));
unique_ptr<int> ptr2 = make_unique<int>(20);  // C++14推荐

*ptr2 = 30;
// unique_ptr<int> ptr3 = ptr2;  // 错误：不能拷贝
unique_ptr<int> ptr3 = move(ptr2);  // 可以移动
// ptr2现在为空

// unique_ptr数组
unique_ptr<int[]> arr(new int[10]);
arr[0] = 1;

// shared_ptr（共享所有权）
shared_ptr<int> sp1 = make_shared<int>(10);
shared_ptr<int> sp2 = sp1;  // 引用计数+1

cout << sp1.use_count() << endl;  // 2
sp2.reset();  // sp2释放，引用计数-1
cout << sp1.use_count() << endl;  // 1

// weak_ptr（弱引用，避免循环引用）
shared_ptr<int> sp = make_shared<int>(10);
weak_ptr<int> wp = sp;  // 不增加引用计数

if (auto locked = wp.lock()) {  // 尝试获取shared_ptr
    cout << *locked << endl;
}

// 自定义删除器
auto deleter = [](int* p) {
    cout << "Custom delete" << endl;
    delete p;
};

unique_ptr<int, decltype(deleter)> ptr(new int(10), deleter);
shared_ptr<int> sptr(new int(20), deleter);
```

### RAII（资源获取即初始化）
```cpp
// 文件管理
class FileHandler {
private:
    FILE* file;
    
public:
    FileHandler(const char* filename, const char* mode) {
        file = fopen(filename, mode);
        if (!file) {
            throw runtime_error("Cannot open file");
        }
    }
    
    ~FileHandler() {
        if (file) {
            fclose(file);
        }
    }
    
    // 禁止拷贝
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
    
    FILE* get() { return file; }
};

// 使用
{
    FileHandler fh("data.txt", "r");
    // 使用文件
}  // 自动关闭文件

// 锁管理
#include <mutex>
mutex mtx;

void threadFunction() {
    lock_guard<mutex> lock(mtx);  // 自动加锁
    // 临界区代码
}  // 自动解锁
```

### 内存泄漏检测
```cpp
// 重载new和delete来追踪
void* operator new(size_t size) {
    void* ptr = malloc(size);
    cout << "Allocated " << size << " bytes at " << ptr << endl;
    return ptr;
}

void operator delete(void* ptr) noexcept {
    cout << "Deallocated at " << ptr << endl;
    free(ptr);
}

// 使用智能指针避免泄漏
void goodFunction() {
    auto ptr = make_unique<int>(10);
    // 自动释放，即使发生异常
}

void badFunction() {
    int* ptr = new int(10);
    // 如果这里发生异常，内存泄漏
    delete ptr;
}
```

### 对象池模式
```cpp
template<typename T>
class ObjectPool {
private:
    vector<unique_ptr<T>> pool;
    vector<T*> available;
    
public:
    T* acquire() {
        if (available.empty()) {
            pool.push_back(make_unique<T>());
            return pool.back().get();
        }
        
        T* obj = available.back();
        available.pop_back();
        return obj;
    }
    
    void release(T* obj) {
        available.push_back(obj);
    }
};
```

---

## 九、STL容器

### vector（动态数组）
```cpp
#include <vector>

// 创建
vector<int> vec;                    // 空vector
vector<int> vec(10);                // 10个元素，默认值0
vector<int> vec(10, 5);             // 10个元素，值都是5
vector<int> vec = {1, 2, 3, 4, 5}; // 初始化列表
vector<int> vec2(vec);              // 拷贝构造

// 添加元素
vec.push_back(10);                  // 尾部添加
vec.emplace_back(20);               // 就地构造（C++11，更高效）

// 访问元素
vec[0] = 1;                         // 下标访问
vec.at(1) = 2;                      // 安全访问（检查边界）
vec.front() = 3;                    // 第一个元素
vec.back() = 4;                     // 最后一个元素

// 删除元素
vec.pop_back();                     // 删除最后一个
vec.erase(vec.begin());             // 删除第一个
vec.erase(vec.begin() + 2);         // 删除索引2
vec.erase(vec.begin(), vec.begin() + 3); // 删除范围
vec.clear();                        // 清空

// 插入元素
vec.insert(vec.begin(), 10);        // 在开头插入
vec.insert(vec.begin() + 2, 20);    // 在索引2插入
vec.insert(vec.end(), 3, 30);       // 在末尾插入3个30

// 大小和容量
vec.size();                         // 元素个数
vec.capacity();                     // 容量
vec.empty();                        // 是否为空
vec.resize(20);                     // 调整大小
vec.reserve(100);                   // 预留容量
vec.shrink_to_fit();                // 释放多余容量（C++11）

// 遍历
for (int i = 0; i < vec.size(); i++) {
    cout << vec[i] << " ";
}

for (auto it = vec.begin(); it != vec.end(); ++it) {
    cout << *it << " ";
}

for (const auto& val : vec) {       // 范围for循环
    cout << val << " ";
}

// 算法
sort(vec.begin(), vec.end());       // 排序
reverse(vec.begin(), vec.end());    // 反转
auto it = find(vec.begin(), vec.end(), 10); // 查找
```

### list（双向链表）
```cpp
#include <list>

list<int> lst = {1, 2, 3, 4, 5};

// 添加元素
lst.push_back(6);
lst.push_front(0);
lst.emplace_back(7);
lst.emplace_front(-1);

// 删除元素
lst.pop_back();
lst.pop_front();
lst.remove(3);                      // 删除所有值为3的元素
lst.remove_if([](int n) { return n > 5; }); // 条件删除

// 访问
lst.front();
lst.back();
// lst[0];  // 错误：不支持随机访问

// 插入
auto it = lst.begin();
advance(it, 2);                     // 移动迭代器
lst.insert(it, 10);

// 操作
lst.sort();                         // 排序
lst.reverse();                      // 反转
lst.unique();                       // 删除连续重复元素

// 合并和拼接
list<int> lst2 = {10, 20, 30};
lst.merge(lst2);                    // 合并已排序列表
lst.splice(lst.begin(), lst2);      // 拼接列表
```

### deque（双端队列）
```cpp
#include <deque>

deque<int> dq = {1, 2, 3, 4, 5};

// 两端操作
dq.push_front(0);
dq.push_back(6);
dq.pop_front();
dq.pop_back();

// 随机访问
dq[0] = 10;
dq.at(1) = 20;

// 其他操作类似vector
dq.insert(dq.begin() + 2, 15);
dq.erase(dq.begin());
```

### stack（栈）
```cpp
#include <stack>

stack<int> stk;

// 操作
stk.push(1);                        // 入栈
stk.push(2);
stk.push(3);

stk.top();                          // 查看栈顶
stk.pop();                          // 出栈

stk.empty();                        // 是否为空
stk.size();                         // 大小

// 遍历（需要弹出元素）
while (!stk.empty()) {
    cout << stk.top() << " ";
    stk.pop();
}
```

### queue（队列）
```cpp
#include <queue>

queue<int> q;

// 操作
q.push(1);                          // 入队
q.push(2);
q.push(3);

q.front();                          // 队首
q.back();                           // 队尾
q.pop();                            // 出队

q.empty();
q.size();
```

### priority_queue（优先队列）
```cpp
#include <queue>

// 默认大顶堆
priority_queue<int> pq;
pq.push(3);
pq.push(1);
pq.push(4);
pq.push(2);

pq.top();                           // 4（最大值）
pq.pop();

// 小顶堆
priority_queue<int, vector<int>, greater<int>> minPq;
minPq.push(3);
minPq.push(1);
minPq.push(4);
minPq.top();                        // 1（最小值）

// 自定义比较
struct Compare {
    bool operator()(int a, int b) {
        return a > b;  // 小顶堆
    }
};
priority_queue<int, vector<int>, Compare> customPq;
```

### set（集合）
```cpp
#include <set>

// 自动排序，元素唯一
set<int> s = {3, 1, 4, 1, 5, 9};    // {1, 3, 4, 5, 9}

// 插入
s.insert(2);
s.emplace(6);
auto [it, success] = s.insert(3);   // C++17：返回迭代器和是否插入成功

// 删除
s.erase(4);                         // 删除值为4的元素
s.erase(s.begin());                 // 删除第一个元素

// 查找
auto it = s.find(5);                // 找到返回迭代器，否则返回end()
if (it != s.end()) {
    cout << "Found: " << *it << endl;
}

s.count(3);                         // 存在返回1，否则0
s.contains(3);                      // C++20：直接返回bool

// 范围查询
auto lower = s.lower_bound(3);      // 第一个>=3的元素
auto upper = s.upper_bound(5);      // 第一个>5的元素

// multiset（允许重复）
multiset<int> ms = {1, 2, 2, 3, 3, 3};
ms.count(3);                        // 3
```

### map（映射）
```cpp
#include <map>

// 键值对，键唯一且自动排序
map<string, int> m;

// 插入
m["apple"] = 1;
m["banana"] = 2;
m.insert({"cherry", 3});
m.insert(make_pair("date", 4));
m.emplace("elderberry", 5);

// 访问
m["apple"];                         // 1
m.at("banana");                     // 2（安全访问）
// m.at("xyz");                     // 抛出异常

// 查找
if (m.find("apple") != m.end()) {
    cout << "Found" << endl;
}

if (m.count("banana")) {
    cout << "Exists" << endl;
}

if (m.contains("cherry")) {         // C++20
    cout << "Exists" << endl;
}

// 删除
m.erase("apple");
m.erase(m.begin());

// 遍历
for (auto& [key, value] : m) {      // C++17结构化绑定
    cout << key << ": " << value << endl;
}

for (auto it = m.begin(); it != m.end(); ++it) {
    cout << it->first << ": " << it->second << endl;
}

// multimap（允许重复键）
multimap<string, int> mm;
mm.insert({"apple", 1});
mm.insert({"apple", 2});
mm.count("apple");                  // 2
```

### unordered_set和unordered_map（哈希表）
```cpp
#include <unordered_set>
#include <unordered_map>

// 无序，平均O(1)查找
unordered_set<int> us = {3, 1, 4, 1, 5};
us.insert(2);
us.find(3);
us.count(4);

// unordered_map
unordered_map<string, int> um;
um["key1"] = 10;
um["key2"] = 20;

// 操作类似set和map，但无序
// 性能通常更好，但无法保证顺序
```

---

## 十、STL算法

### 非修改序列算法
```cpp
#include <algorithm>

vector<int> vec = {1, 2, 3, 4, 5, 3, 2, 1};

// 查找
auto it = find(vec.begin(), vec.end(), 3);
if (it != vec.end()) {
    cout << "Found at: " << distance(vec.begin(), it) << endl;
}

// 条件查找
auto it2 = find_if(vec.begin(), vec.end(), [](int n) {
    return n > 3;
});

// 计数
int cnt = count(vec.begin(), vec.end(), 3);        // 2
int cnt2 = count_if(vec.begin(), vec.end(), [](int n) {
    return n % 2 == 0;
});

// 遍历
for_each(vec.begin(), vec.end(), [](int n) {
    cout << n << " ";
});

// 比较
vector<int> vec2 = {1, 2, 3, 4, 5, 3, 2, 1};
bool equal = equal(vec.begin(), vec.end(), vec2.begin());

// 搜索子序列
vector<int> sub = {3, 4, 5};
auto pos = search(vec.begin(), vec.end(), sub.begin(), sub.end());

// 最大最小
auto minIt = min_element(vec.begin(), vec.end());
auto maxIt = max_element(vec.begin(), vec.end());
auto [min, max] = minmax_element(vec.begin(), vec.end()); // C++11
```

### 修改序列算法
```cpp
vector<int> vec = {1, 2, 3, 4, 5};

// 复制
vector<int> dest(5);
copy(vec.begin(), vec.end(), dest.begin());

copy_if(vec.begin(), vec.end(), dest.begin(), [](int n) {
    return n % 2 == 0;
});

// 移动
vector<int> dest2(5);
move(vec.begin(), vec.end(), dest2.begin());

// 填充
fill(vec.begin(), vec.end(), 0);           // 全部填充0
fill_n(vec.begin(), 3, 5);                 // 前3个填充5

// 生成
generate(vec.begin(), vec.end(), []() {
    return rand() % 100;
});

// 转换
vector<int> squares(5);
transform(vec.begin(), vec.end(), squares.begin(), [](int n) {
    return n * n;
});

// 两个序列转换
vector<int> a = {1, 2, 3};
vector<int> b = {4, 5, 6};
vector<int> c(3);
transform(a.begin(), a.end(), b.begin(), c.begin(), [](int x, int y) {
    return x + y;
});

// 替换
replace(vec.begin(), vec.end(), 0, -1);    // 0替换为-1
replace_if(vec.begin(), vec.end(), [](int n) {
    return n < 0;
}, 0);

// 删除
auto newEnd = remove(vec.begin(), vec.end(), 3);
vec.erase(newEnd, vec.end());              // 真正删除

remove_if(vec.begin(), vec.end(), [](int n) {
    return n % 2 == 0;
});

// 去重（需要先排序）
sort(vec.begin(), vec.end());
auto last = unique(vec.begin(), vec.end());
vec.erase(last, vec.end());

// 反转
reverse(vec.begin(), vec.end());

// 旋转
rotate(vec.begin(), vec.begin() + 2, vec.end());
```

### 排序算法
```cpp
vector<int> vec = {5, 2, 8, 1, 9, 3};

// 排序
sort(vec.begin(), vec.end());              // 升序
sort(vec.begin(), vec.end(), greater<int>()); // 降序

// 自定义比较
sort(vec.begin(), vec.end(), [](int a, int b) {
    return abs(a) < abs(b);
});

// 稳定排序
stable_sort(vec.begin(), vec.end());

// 部分排序
partial_sort(vec.begin(), vec.begin() + 3, vec.end()); // 前3个有序

// 第n个元素
nth_element(vec.begin(), vec.begin() + 3, vec.end()); // 第4个元素到正确位置

// 二分查找（需要已排序）
sort(vec.begin(), vec.end());
bool found = binary_search(vec.begin(), vec.end(), 5);

auto lower = lower_bound(vec.begin(), vec.end(), 5);
auto upper = upper_bound(vec.begin(), vec.end(), 5);

// 归并
vector<int> v1 = {1, 3, 5};
vector<int> v2 = {2, 4, 6};
vector<int> result(6);
merge(v1.begin(), v1.end(), v2.begin(), v2.end(), result.begin());
```

### 数值算法
```cpp
#include <numeric>

vector<int> vec = {1, 2, 3, 4, 5};

// 累加
int sum = accumulate(vec.begin(), vec.end(), 0);

// 自定义操作
int product = accumulate(vec.begin(), vec.end(), 1, [](int a, int b) {
    return a * b;
});

// 相邻差
vector<int> diff(5);
adjacent_difference(vec.begin(), vec.end(), diff.begin());

// 部分和
vector<int> partialSum(5);
partial_sum(vec.begin(), vec.end(), partialSum.begin());

// 内积
vector<int> v2 = {1, 1, 1, 1, 1};
int dot = inner_product(vec.begin(), vec.end(), v2.begin(), 0);

// iota（生成序列）
vector<int> seq(10);
iota(seq.begin(), seq.end(), 1);           // 1, 2, 3, ..., 10
```

---

## 十一、模板

### 函数模板
```cpp
// 基本函数模板
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// 使用
int maxInt = max(10, 20);                  // T = int
double maxDouble = max(3.14, 2.71);        // T = double
max<int>(10, 20);                          // 显式指定类型

// 多个模板参数
template<typename T1, typename T2>
auto add(T1 a, T2 b) -> decltype(a + b) {  // C++11
    return a + b;
}

// C++14: 返回类型自动推导
template<typename T1, typename T2>
auto add(T1 a, T2 b) {
    return a + b;
}

// 模板特化
template<typename T>
T absolute(T value) {
    return value < 0 ? -value : value;
}

// 全特化
template<>
string absolute<string>(string value) {
    return value;  // 字符串不需要绝对值
}

// 函数模板重载
template<typename T>
void print(T value) {
    cout << value << endl;
}

template<typename T>
void print(T* ptr) {
    cout << *ptr << endl;
}
```

### 类模板
```cpp
// 基本类模板
template<typename T>
class Box {
private:
    T value;
    
public:
    Box(T v) : value(v) {}
    
    T getValue() const {
        return value;
    }
    
    void setValue(T v) {
        value = v;
    }
};

// 使用
Box<int> intBox(10);
Box<string> strBox("Hello");

// 多个模板参数
template<typename K, typename V>
class Pair {
private:
    K key;
    V value;
    
public:
    Pair(K k, V v) : key(k), value(v) {}
    
    K getKey() const { return key; }
    V getValue() const { return value; }
};

Pair<string, int> p("age", 25);

// 默认模板参数
template<typename T = int>
class Container {
    T data;
public:
    Container(T d) : data(d) {}
};

Container<> c1(10);        // 使用默认int
Container<double> c2(3.14);

// 类模板特化
template<typename T>
class Storage {
    T data;
public:
    void set(T value) { data = value; }
    T get() { return data; }
};

// 全特化
template<>
class Storage<bool> {
    unsigned char data;
public:
    void set(bool value) { data = value ? 1 : 0; }
    bool get() { return data != 0; }
};

// 部分特化
template<typename T>
class Storage<T*> {
    T* data;
public:
    void set(T* value) { data = value; }
    T* get() { return data; }
};
```

### 可变参数模板（C++11）
```cpp
// 递归展开
template<typename T>
void print(T value) {
    cout << value << endl;
}

template<typename T, typename... Args>
void print(T first, Args... args) {
    cout << first << ", ";
    print(args...);
}

print(1, 2.5, "hello", 'A');

// sizeof...运算符
template<typename... Args>
void printCount(Args... args) {
    cout << "参数个数: " << sizeof...(args) << endl;
}

// 折叠表达式（C++17）
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);           // 一元右折叠
}

template<typename... Args>
auto sum2(Args... args) {
    return (... + args);           // 一元左折叠
}

template<typename... Args>
void printAll(Args... args) {
    (cout << ... << args) << endl; // 折叠输出
}

// 完美转发
template<typename T>
void wrapper(T&& arg) {
    realFunction(std::forward<T>(arg));
}

template<typename... Args>
void wrapper(Args&&... args) {
    realFunction(std::forward<Args>(args)...);
}
```

### 模板别名（C++11）
```cpp
// 类型别名
template<typename T>
using Vec = vector<T>;

Vec<int> numbers;  // 等价于 vector<int>

// 模板化的类型别名
template<typename T>
using Ptr = shared_ptr<T>;

Ptr<int> p = make_shared<int>(10);
```

### 概念（C++20）
```cpp
// 定义概念
template<typename T>
concept Numeric = is_arithmetic_v<T>;

template<typename T>
concept Printable = requires(T t) {
    { cout << t } -> same_as<ostream&>;
};

// 使用概念约束模板
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// requires子句
template<typename T>
requires Numeric<T>
T multiply(T a, T b) {
    return a * b;
}

// 简写函数模板
auto divide(Numeric auto a, Numeric auto b) {
    return a / b;
}
```

---

## 十二、现代C++特性

### auto类型推导（C++11）
```cpp
// 自动类型推导
auto x = 10;                    // int
auto y = 3.14;                  // double
auto s = "hello";               // const char*
auto vec = vector<int>{1, 2, 3};

// 与模板结合
vector<int> numbers;
auto it = numbers.begin();      // vector<int>::iterator

// 函数返回值
auto getValue() {
    return 42;
}

// 复杂类型简化
map<string, vector<int>> data;
for (auto& [key, value] : data) {  // C++17
    // 使用key和value
}

// decltype
int x = 10;
decltype(x) y = 20;             // y是int类型

// decltype(auto)（C++14）
decltype(auto) func() {
    int x = 10;
    return x;                    // 返回int，不是int&
}
```

### 范围for循环（C++11）
```cpp
vector<int> vec = {1, 2, 3, 4, 5};

// 基本用法
for (int n : vec) {
    cout << n << " ";
}

// 引用（可修改）
for (int& n : vec) {
    n *= 2;
}

// 常量引用（只读，避免拷贝）
for (const auto& n : vec) {
    cout << n << " ";
}

// 初始化语句（C++20）
for (auto vec = getData(); auto& n : vec) {
    cout << n << " ";
}
```

### 统一初始化（C++11）
```cpp
// 所有类型统一使用{}初始化
int x{10};
double y{3.14};
string s{"hello"};

// 容器初始化
vector<int> vec{1, 2, 3, 4, 5};
map<string, int> m{{"a", 1}, {"b", 2}};

// 防止窄化转换
int a = 3.14;       // OK，但会截断
// int b{3.14};     // 错误：窄化转换

// 初始化列表
class Point {
public:
    int x, y;
    Point(int x, int y) : x(x), y(y) {}
};

Point p{10, 20};

// initializer_list
void func(initializer_list<int> list) {
    for (auto n : list) {
        cout << n << " ";
    }
}

func({1, 2, 3, 4, 5});
```

### nullptr（C++11）
```cpp
// 替代NULL和0
int* ptr = nullptr;

void func(int x) { cout << "int" << endl; }
void func(int* ptr) { cout << "pointer" << endl; }

func(0);            // 调用int版本（可能不是想要的）
func(NULL);         // 可能有歧义
func(nullptr);      // 调用pointer版本，明确
```

### 强枚举类型（C++11）
```cpp
// 传统枚举的问题
enum Color { Red, Green, Blue };
enum TrafficLight { Red, Yellow, Green };  // 错误：重复定义

// 强枚举类型（enum class）
enum class Color { Red, Green, Blue };
enum class TrafficLight { Red, Yellow, Green };  // OK

Color c = Color::Red;
TrafficLight t = TrafficLight::Red;

// 不能隐式转换
// int x = Color::Red;  // 错误
int x = static_cast<int>(Color::Red);  // OK

// 指定底层类型
enum class Status : uint8_t {
    OK = 0,
    Error = 1,
    Pending = 2
};
```

### 委托构造函数（C++11）
```cpp
class Person {
private:
    string name;
    int age;
    string email;
    
public:
    // 主构造函数
    Person(string n, int a, string e)
        : name(n), age(a), email(e) {}
    
    // 委托给主构造函数
    Person(string n, int a)
        : Person(n, a, "") {}
    
    Person(string n)
        : Person(n, 0, "") {}
    
    Person()
        : Person("", 0, "") {}
};
```

### 继承构造函数（C++11）
```cpp
class Base {
public:
    Base(int x) {}
    Base(int x, int y) {}
};

class Derived : public Base {
public:
    using Base::Base;  // 继承所有构造函数
    
    // 可以添加自己的成员
    string name;
};

Derived d1(10);        // 调用Base(int)
Derived d2(10, 20);    // 调用Base(int, int)
```

### override和final（C++11）
```cpp
class Base {
public:
    virtual void func1() {}
    virtual void func2() {}
    virtual void func3() final {}  // 不能被重写
};

class Derived : public Base {
public:
    void func1() override {}       // 明确表示重写
    // void func2() const override {}  // 错误：签名不匹配
    // void func3() override {}    // 错误：final函数不能重写
    
    void func4() {}                // 新函数，不是重写
};

// final类（不能被继承）
class FinalClass final {
    // ...
};

// class CannotInherit : public FinalClass {};  // 错误
```

### 默认和删除函数（C++11）
```cpp
class MyClass {
public:
    // 显式使用默认实现
    MyClass() = default;
    MyClass(const MyClass&) = default;
    MyClass& operator=(const MyClass&) = default;
    ~MyClass() = default;
    
    // 禁用某些函数
    MyClass(MyClass&&) = delete;
    MyClass& operator=(MyClass&&) = delete;
};

// 禁用特定重载
void func(int x) {}
void func(double x) = delete;  // 禁止double调用

func(10);       // OK
// func(3.14);  // 错误：函数已删除
```

### constexpr（C++11/14/17）
```cpp
// 编译期常量表达式
constexpr int square(int x) {
    return x * x;
}

constexpr int val = square(5);  // 编译期计算
int arr[square(10)];            // 可用于数组大小

// C++14: 允许更复杂的constexpr函数
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// C++17: constexpr if
template<typename T>
auto getValue(T t) {
    if constexpr (is_pointer_v<T>) {
        return *t;
    } else {
        return t;
    }
}

// constexpr变量
constexpr double PI = 3.14159;
constexpr int MAX_SIZE = 100;
```

### 结构化绑定（C++17）
```cpp
// 解构pair
pair<int, string> p{1, "hello"};
auto [id, name] = p;
cout << id << ", " << name << endl;

// 解构tuple
tuple<int, double, string> t{1, 3.14, "world"};
auto [a, b, c] = t;

// 解构数组
int arr[] = {1, 2, 3};
auto [x, y, z] = arr;

// 解构结构体
struct Point {
    int x, y;
};
Point pt{10, 20};
auto [px, py] = pt;

// 在循环中使用
map<string, int> m{{"a", 1}, {"b", 2}};
for (auto& [key, value] : m) {
    cout << key << ": " << value << endl;
}
```

### if和switch初始化器（C++17）
```cpp
// if with initializer
if (auto it = m.find("key"); it != m.end()) {
    cout << it->second << endl;
    // it只在if作用域内有效
}

// switch with initializer
switch (auto status = getStatus(); status) {
    case Status::OK:
        break;
    case Status::Error:
        break;
}
```

### std::optional（C++17）
```cpp
#include <optional>

// 可能有值，也可能没有
optional<int> divide(int a, int b) {
    if (b == 0) {
        return nullopt;  // 无值
    }
    return a / b;
}

// 使用
auto result = divide(10, 2);
if (result.has_value()) {
    cout << "结果: " << result.value() << endl;
}

// 或者
if (result) {
    cout << "结果: " << *result << endl;
}

// value_or提供默认值
int val = result.value_or(0);

// emplace
optional<string> opt;
opt.emplace("Hello");  // 就地构造
```

### std::variant（C++17）
```cpp
#include <variant>

// 类型安全的union
variant<int, double, string> v;

v = 10;                // 存储int
v = 3.14;              // 存储double
v = "hello";           // 存储string

// 访问
if (holds_alternative<int>(v)) {
    cout << get<int>(v) << endl;
}

// 访问（带异常）
try {
    cout << get<double>(v) << endl;
} catch (bad_variant_access&) {
    cout << "类型不匹配" << endl;
}

// 访问（返回指针）
if (auto p = get_if<string>(&v)) {
    cout << *p << endl;
}

// visit访问器
visit([](auto&& arg) {
    using T = decay_t<decltype(arg)>;
    if constexpr (is_same_v<T, int>) {
        cout << "int: " << arg << endl;
    } else if constexpr (is_same_v<T, double>) {
        cout << "double: " << arg << endl;
    } else {
        cout << "string: " << arg << endl;
    }
}, v);
```

### std::any（C++17）
```cpp
#include <any>

// 可以存储任何类型
any a = 10;
a = 3.14;
a = string("hello");

// 访问
if (a.has_value()) {
    if (a.type() == typeid(string)) {
        cout << any_cast<string>(a) << endl;
    }
}

// 修改
a.emplace<vector<int>>({1, 2, 3});

// 重置
a.reset();
```

### std::string_view（C++17）
```cpp
#include <string_view>

// 非拥有的字符串视图，避免拷贝
void printString(string_view sv) {
    cout << sv << endl;
}

string s = "Hello, World!";
printString(s);                    // 不拷贝
printString("Literal");            // 不拷贝
printString(s.substr(0, 5));       // 避免临时对象

// 操作
string_view sv = "Hello, World!";
sv.substr(0, 5);                   // "Hello"
sv.remove_prefix(7);               // "World!"
sv.remove_suffix(1);               // "World"
```

### 文件系统库（C++17）
```cpp
#include <filesystem>
namespace fs = std::filesystem;

// 路径操作
fs::path p = "/home/user/file.txt";
cout << p.filename() << endl;      // "file.txt"
cout << p.extension() << endl;     // ".txt"
cout << p.parent_path() << endl;   // "/home/user"

// 检查文件
if (fs::exists(p)) {
    cout << "文件存在" << endl;
}

if (fs::is_regular_file(p)) {
    cout << "是普通文件" << endl;
}

// 文件大小
auto size = fs::file_size(p);

// 遍历目录
for (auto& entry : fs::directory_iterator("/home/user")) {
    cout << entry.path() << endl;
}

// 递归遍历
for (auto& entry : fs::recursive_directory_iterator("/home/user")) {
    if (entry.is_regular_file()) {
        cout << entry.path() << endl;
    }
}

// 创建目录
fs::create_directory("new_dir");
fs::create_directories("path/to/nested/dir");

// 删除
fs::remove("file.txt");
fs::remove_all("directory");

// 复制
fs::copy("src.txt", "dst.txt");
```

### 并发库（C++11）
```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>

// 线程
void threadFunc(int x) {
    cout << "Thread: " << x << endl;
}

thread t1(threadFunc, 10);
t1.join();  // 等待线程结束

thread t2([]() {
    cout << "Lambda thread" << endl;
});
t2.detach();  // 分离线程

// 互斥锁
mutex mtx;

void criticalSection() {
    mtx.lock();
    // 临界区代码
    mtx.unlock();
}

// RAII锁管理
void safeCriticalSection() {
    lock_guard<mutex> lock(mtx);
    // 临界区代码
}  // 自动解锁

// 条件变量
mutex m;
condition_variable cv;
bool ready = false;

void waitForSignal() {
    unique_lock<mutex> lock(m);
    cv.wait(lock, []{ return ready; });
    // 继续执行
}

void sendSignal() {
    {
        lock_guard<mutex> lock(m);
        ready = true;
    }
    cv.notify_one();  // 或 notify_all()
}

// 原子操作
atomic<int> counter{0};

void increment() {
    counter++;  // 原子操作
}

// future和promise
promise<int> prom;
future<int> fut = prom.get_future();

thread t([](promise<int> p) {
    this_thread::sleep_for(chrono::seconds(1));
    p.set_value(42);
}, move(prom));

int result = fut.get();  // 阻塞等待结果
t.join();

// async异步任务
auto future1 = async(launch::async, []() {
    return 42;
});

auto future2 = async(launch::deferred, []() {
    return 100;  // 延迟执行
});

cout << future1.get() << endl;
```

### 智能指针（C++11）
```cpp
#include <memory>

// unique_ptr
auto ptr1 = make_unique<int>(10);
auto ptr2 = make_unique<int[]>(10);  // 数组

// shared_ptr
auto sp1 = make_shared<int>(10);
auto sp2 = sp1;  // 引用计数+1
cout << sp1.use_count() << endl;  // 2

// weak_ptr
weak_ptr<int> wp = sp1;
if (auto locked = wp.lock()) {
    cout << *locked << endl;
}

// 自定义删除器
auto deleter = [](int* p) {
    cout << "删除" << endl;
    delete p;
};
shared_ptr<int> sp(new int(10), deleter);

// enable_shared_from_this
class MyClass : public enable_shared_from_this<MyClass> {
public:
    shared_ptr<MyClass> getPtr() {
        return shared_from_this();
    }
};
```

### 移动语义（C++11）
```cpp
// 移动构造函数
class String {
    char* data;
    size_t size;
    
public:
    // 移动构造
    String(String&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    
    // 移动赋值
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};

// std::move
String s1 = "Hello";
String s2 = std::move(s1);  // s1资源转移到s2

// 返回值优化（RVO）自动使用移动
String createString() {
    String s = "Temp";
    return s;  // 自动移动，不拷贝
}
```

### 完美转发（C++11）
```cpp
// 转发引用（万能引用）
template<typename T>
void wrapper(T&& arg) {
    // 保持参数的值类别
    realFunction(std::forward<T>(arg));
}

// 可变参数完美转发
template<typename... Args>
void wrapper(Args&&... args) {
    realFunction(std::forward<Args>(args)...);
}

// emplace系列函数使用完美转发
vector<pair<int, string>> vec;
vec.emplace_back(1, "hello");  // 就地构造，不拷贝
```

### 用户定义字面量（C++11）
```cpp
// 定义字面量后缀
constexpr long double operator"" _km(long double x) {
    return x * 1000;
}

constexpr long double operator"" _m(long double x) {
    return x;
}

auto distance1 = 5.0_km;  // 5000米
auto distance2 = 100.0_m; // 100米

// 字符串字面量
string operator"" _s(const char* str, size_t len) {
    return string(str, len);
}

auto s = "hello"_s;  // string类型

// 标准库提供的字面量
using namespace std::literals;
auto str = "hello"s;              // string
auto duration = 10s;              // chrono::seconds
auto duration2 = 500ms;           // chrono::milliseconds
```

### 属性（C++11/14/17）
```cpp
// [[noreturn]]
[[noreturn]] void terminate() {
    exit(1);
}

// [[deprecated]]
[[deprecated("使用newFunction代替")]]
void oldFunction() {}

// [[nodiscard]]（C++17）
[[nodiscard]] int getValue() {
    return 42;
}
// getValue();  // 警告：忽略返回值

// [[maybe_unused]]（C++17）
void func([[maybe_unused]] int x) {
    // x可能不使用
}

// [[fallthrough]]（C++17）
switch (value) {
    case 1:
        doSomething();
        [[fallthrough]];
    case 2:
        doMore();
        break;
}
```

### 三路比较运算符（C++20）
```cpp
#include <compare>

class Point {
    int x, y;
    
public:
    Point(int x, int y) : x(x), y(y) {}
    
    // 默认实现
    auto operator<=>(const Point&) const = default;
    
    // 或自定义
    strong_ordering operator<=>(const Point& other) const {
        if (auto cmp = x <=> other.x; cmp != 0)
            return cmp;
        return y <=> other.y;
    }
};

Point p1{1, 2}, p2{3, 4};
bool less = p1 < p2;
bool equal = p1 == p2;
bool greater = p1 > p2;

// 比较类别
// strong_ordering: 强序（可替换）
// weak_ordering: 弱序（等价但不可替换）
// partial_ordering: 偏序（可能不可比）
```

### 协程（C++20）
```cpp
#include <coroutine>

// 简单的Generator
template<typename T>
struct Generator {
    struct promise_type {
        T value;
        
        Generator get_return_object() {
            return Generator{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }
        
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() {}
        
        std::suspend_always yield_value(T val) {
            value = val;
            return {};
        }
    };
    
    std::coroutine_handle<promise_type> handle;
    
    ~Generator() {
        if (handle) handle.destroy();
    }
    
    bool next() {
        handle.resume();
        return !handle.done();
    }
    
    T value() {
        return handle.promise().value;
    }
};

// 使用协程
Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        auto temp = a;
        a = b;
        b = temp + b;
    }
}

auto gen = fibonacci();
for (int i = 0; i < 10; ++i) {
    gen.next();
    cout << gen.value() << " ";
}
```

### 模块（C++20）
```cpp
// math.cppm (模块接口文件)
export module math;

export int add(int a, int b) {
    return a + b;
}

export int multiply(int a, int b) {
    return a * b;
}

// 私有函数（不导出）
int helper() {
    return 42;
}

// main.cpp (使用模块)
import math;

int main() {
    int result = add(10, 20);
    // int x = helper();  // 错误：helper未导出
}
```

### Ranges库（C++20）
```cpp
#include <ranges>
namespace views = std::views;

vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// 过滤和转换
auto result = vec 
    | views::filter([](int n) { return n % 2 == 0; })
    | views::transform([](int n) { return n * n; });

for (int n : result) {
    cout << n << " ";  // 4 16 36 64 100
}

// 惰性求值
auto view = views::iota(1)  // 无限序列
    | views::take(10)        // 取前10个
    | views::filter([](int n) { return n % 2 == 0; });

// 更多操作
auto v1 = vec | views::reverse;
auto v2 = vec | views::drop(3);     // 跳过前3个
auto v3 = vec | views::take(5);     // 取前5个
auto v4 = vec | views::split(5);    // 按值分割
```

---

## 附录：常用技巧

### 1. RAII惯用法
```cpp
// 资源管理
class Resource {
    int* data;
public:
    Resource() : data(new int[100]) {}
    ~Resource() { delete[] data; }
    
    // 禁止拷贝
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    
    // 允许移动
    Resource(Resource&& other) noexcept : data(other.data) {
        other.data = nullptr;
    }
};
```

### 2. Pimpl惯用法
```cpp
// Widget.h
class Widget {
    class Impl;
    unique_ptr<Impl> pImpl;
    
public:
    Widget();
    ~Widget();
    void doSomething();
};

// Widget.cpp
class Widget::Impl {
public:
    void doSomething() { /* 实现 */ }
};

Widget::Widget() : pImpl(make_unique<Impl>()) {}
Widget::~Widget() = default;
void Widget::doSomething() { pImpl->doSomething(); }
```

### 3. 单例模式
```cpp
class Singleton {
private:
    Singleton() {}
    
public:
    static Singleton& getInstance() {
        static Singleton instance;  // C++11线程安全
        return instance;
    }
    
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
};
```

### 4. 类型萃取
```cpp
#include <type_traits>

template<typename T>
void process(T value) {
    if constexpr (is_integral_v<T>) {
        // 整数类型处理
    } else if constexpr (is_floating_point_v<T>) {
        // 浮点类型处理
    } else if constexpr (is_pointer_v<T>) {
        // 指针类型处理
    }
}

// SFINAE
template<typename T>
enable_if_t<is_arithmetic_v<T>, T>
add(T a, T b) {
    return a + b;
}
```

### 5. 调试技巧
```cpp
// 断言
#include <cassert>
assert(x > 0);  // Debug模式检查

// static_assert（编译期）
static_assert(sizeof(int) == 4, "int必须是4字节");

// 打印类型
template<typename T>
void printType(T&&) {
    cout << __PRETTY_FUNCTION__ << endl;  // GCC/Clang
    // cout << __FUNCSIG__ << endl;       // MSVC
}

// 条件编译
#ifdef DEBUG
    cout << "调试信息" << endl;
#endif
```