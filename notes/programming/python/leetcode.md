### 1 两数之和
给定一个整数数组 nums 和一个目标值 target，请在数组中找出两个数，它们的和等于 target，并返回它们的下标。每种输入只有一个解，且不能使用同一个元素两次，可以按任意顺序返回答案。
**示例**：
> 输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 创建一个哈希表，用于存储已遍历的元素及其索引
        seen = {}
        for i, num in enumerate(nums):
            # 计算目标差值
            complement = target - num
            # 如果差值已经在哈希表中，说明找到了结果
            if complement in seen:
                return [seen[complement], i]
            # 否则将当前元素存入哈希表
            seen[num] = i
        return []  # 如果没有找到符合条件的答案

```

### 2 两数相加
给给定两个非空链表，表示两个非负整数，链表中的每个节点存储一个数字，且数字按逆序排列。请将两个数相加，并以链表形式返回和，假设两个链表的数字都不以 0 开头（除了数字 0）。

**示例**：
> 输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```python
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        carry = 0  # 存储进位
        dummy_head = ListNode()  # 创建一个虚拟头节点，方便操作
        current = dummy_head  # 用于构建结果链表

        # 逐位相加，直到两个链表都遍历完
        while l1 or l2 or carry:
            # 获取当前节点的值，如果当前链表为空，则取 0
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0

            # 计算当前位的和和进位
            total = val1 + val2 + carry
            carry = total // 10  # 计算新的进位
            current.next = ListNode(total % 10)  # 创建新节点存储当前位的值
            current = current.next  # 移动指针

            # 移动 l1 和 l2 指针
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        # 返回虚拟头节点的下一个节点，即结果链表的头节点
        return dummy_head.next

# 辅助函数：创建一个链表
def create_linked_list(nums):
    head = ListNode(nums[0])
    current = head
    for num in nums[1:]:
        current.next = ListNode(num)
        current = current.next
    return head

# 辅助函数：打印链表
def print_linked_list(head):
    current = head
    while current:
        print(current.val, end=" -> ")
        current = current.next
    print("None")

```

### 3 无重复字符串的最长子串
给定一个字符串 s ，请找出其中不含有重复字符的最长子串的长度。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = set()  # 存储当前窗口中的字符
        left = 0  # 左指针
        max_len = 0  # 最长子串的长度
        
        for right in range(len(s)):  # 右指针遍历整个字符串
            # 如果右指针指向的字符在窗口中出现过，移动左指针
            while s[right] in seen:
                seen.remove(s[left])  # 移除左指针指向的字符
                left += 1  # 向右移动左指针
            
            seen.add(s[right])  # 将当前字符加入窗口
            max_len = max(max_len, right - left + 1)  # 更新最长子串长度
        
        return max_len

```

### 4 寻找两个正序数组的中位数
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。算法的时间复杂度应该为 O(log (m+n)) 。

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        A, B = nums1, nums2
        m, n = len(A), len(B)

        # 保证 A 是较短的数组
        if m > n:
            A, B, m, n = B, A, n, m

        total = m + n
        half = total // 2

        left, right = 0, m
        while True:
            i = (left + right) // 2  # A 的切分点
            j = half - i             # B 的切分点

            A_left  = A[i - 1] if i - 1 >= 0 else float('-inf')
            A_right = A[i] if i < m else float('inf')
            B_left  = B[j - 1] if j - 1 >= 0 else float('-inf')
            B_right = B[j] if j < n else float('inf')

            if A_left <= B_right and B_left <= A_right:
                if total % 2:  # 奇数
                    return float(min(A_right, B_right))
                return (max(A_left, B_left) + min(A_right, B_right)) / 2.0
            elif A_left > B_right:
                right = i - 1
            else:
                left = i + 1

```


### 5 最长回文子串
给你一个字符串 s，找到 s 中最长的 回文 子串。
**示例**：
> 输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s or len(s) < 1:
            return ""

        start = 0
        end = 0

        def expandAroundCenter(left: int, right: int) -> int:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1  # 回文长度

        for i in range(len(s)):
            len1 = expandAroundCenter(i, i)      # 奇数长度回文
            len2 = expandAroundCenter(i, i + 1)  # 偶数长度回文
            max_len = max(len1, len2)
            if max_len > end - start:
                # 更新 start 和 end
                start = i - (max_len - 1) // 2
                end = i + max_len // 2

        return s[start:end+1]

```

### 6 Z 字形变换
将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：
```
P   A   H   N
A P L S I I G
Y   I   R
```
之后，输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。

```python
# Completed independently
class SolZigzagNaive:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        str_lst = ["" for _ in range(numRows)]
        s_i = 0
        if numRows == 2:
            for i in range(len(s)):
                if i % 2:
                    str_lst[1] += s[s_i]
                    s_i += 1
                else:
                    str_lst[0] += s[s_i]
                    s_i += 1
        while s_i < len(s):
            str_lst[0] += s[s_i]
            s_i += 1
            i = 1
            corner = False
            while i:
                if s_i >= len(s):
                    break
                if i == numRows:
                    i -= 2
                    corner = True
                if corner:

                    str_lst[i] += s[s_i]
                    s_i += 1
                    i -= 1
                else:
                    str_lst[i] += s[s_i]
                    s_i += 1
                    i += 1

        return "".join(str_lst)

# AI
class Solution:
    def convert(self, s: str, numRows: int) -> str:
         # 特殊情况：只有一行或长度太短，不需要变化
        if numRows == 1 or numRows >= len(s):
            return s

        # 用于存储每一行的字符
        rows = [''] * numRows
        curRow = 0  # 当前行
        direction = -1  # 行走方向（将在需要时反转）

        for char in s:
            rows[curRow] += char  # 把字符放入当前行

            # 到达顶端或底部时改变方向
            if curRow == 0 or curRow == numRows - 1:
                direction *= -1

            curRow += direction  # 向下一行或上一行移动

        # 将所有行拼接成结果
        return ''.join(rows)

```

### 7 整数反转
给定一个 32 位有符号整数 `x`，返回其数字反转后的结果。
若反转后结果超出 32 位有符号整数范围 [-2³¹, 2³¹−1]，返回 0。
假设不能使用 64 位整数类型。

```python
class Solution:
    def reverse(self, x: int) -> int:
        sign = -1 if x < 0 else 1
        rev = int(str(abs(x))[::-1]) * sign
        return rev if -2**31 <= rev <= 2**31 - 1 else 0
```

### 8 字符串转换整数 (atoi)
实现函数 `myAtoi(string s)`，将字符串转换为 32 位有符号整数，处理流程如下：
1. **跳过前导空格**。
2. **识别符号**：若下一个字符是 `'+'` 或 `'-'`，记录符号；否则默认为正数。
3. **读取数字**：从第一个数字开始连续读取，忽略前置零，直到遇到非数字或字符串结束；若没有数字则返回 0。
4. **范围截断**：将结果限制在区间
   [-2³¹, 2³¹ − 1]，超出则截断到边界值。
5. 返回最终整数。

```python
class Solution:
    def myAtoi(self, s: str) -> int:
        digits = set("0123456789")
        s = s.strip()  # 去掉前后空格
        if not s:
            return 0

        sign = 1
        res = ""
        i = 0

        # 处理符号
        if s[0] == "-":
            sign = -1
            i = 1
        elif s[0] == "+":
            i = 1

        # 提取数字部分
        while i < len(s) and s[i] in digits:
            res += s[i]
            i += 1

        if not res:  # 没有数字
            return 0

        num = int(res) * sign

        # 限制范围
        INT_MIN = -2**31
        INT_MAX = 2**31 - 1

        if num < INT_MIN:
            return INT_MIN
        if num > INT_MAX:
            return INT_MAX
        return num

```

### 9 两数之和
判断模式串 `p` 是否能完整匹配整个字符串 `s`，不能只匹配一部分。

**支持的规则：**

* `.` ：匹配任意一个字符
* `*` ：匹配它前面那个字符的零次或多次

```python
# DP 动态规划
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True  # 空串匹配空模式

        # 预处理 pattern 里可能开头是 a*, a*b*, a*b*c* 这种
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*':
                    # 情况一：当作 0 次（删掉 前一个字符+*）
                    dp[i][j] = dp[i][j - 2]
                    # 情况二：前一个字符等于当前字符（或是 .）
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]

        return dp[m][n]

```

### 1 两数之和
给定一个整数数组 nums 和一个目标值 target，请在数组中找出两个数，它们的和等于 target，并返回它们的下标。每种输入只有一个解，且不能使用同一个元素两次，可以按任意顺序返回答案。
**示例**：
> 输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 创建一个哈希表，用于存储已遍历的元素及其索引
        seen = {}
        for i, num in enumerate(nums):
            # 计算目标差值
            complement = target - num
            # 如果差值已经在哈希表中，说明找到了结果
            if complement in seen:
                return [seen[complement], i]
            # 否则将当前元素存入哈希表
            seen[num] = i
        return []  # 如果没有找到符合条件的答案

```

### 1 两数之和
给定一个整数数组 nums 和一个目标值 target，请在数组中找出两个数，它们的和等于 target，并返回它们的下标。每种输入只有一个解，且不能使用同一个元素两次，可以按任意顺序返回答案。
**示例**：
> 输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 创建一个哈希表，用于存储已遍历的元素及其索引
        seen = {}
        for i, num in enumerate(nums):
            # 计算目标差值
            complement = target - num
            # 如果差值已经在哈希表中，说明找到了结果
            if complement in seen:
                return [seen[complement], i]
            # 否则将当前元素存入哈希表
            seen[num] = i
        return []  # 如果没有找到符合条件的答案

```

### 1 两数之和
给定一个整数数组 nums 和一个目标值 target，请在数组中找出两个数，它们的和等于 target，并返回它们的下标。每种输入只有一个解，且不能使用同一个元素两次，可以按任意顺序返回答案。
**示例**：
> 输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 创建一个哈希表，用于存储已遍历的元素及其索引
        seen = {}
        for i, num in enumerate(nums):
            # 计算目标差值
            complement = target - num
            # 如果差值已经在哈希表中，说明找到了结果
            if complement in seen:
                return [seen[complement], i]
            # 否则将当前元素存入哈希表
            seen[num] = i
        return []  # 如果没有找到符合条件的答案

```

### 1 两数之和
给定一个整数数组 nums 和一个目标值 target，请在数组中找出两个数，它们的和等于 target，并返回它们的下标。每种输入只有一个解，且不能使用同一个元素两次，可以按任意顺序返回答案。
**示例**：
> 输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 创建一个哈希表，用于存储已遍历的元素及其索引
        seen = {}
        for i, num in enumerate(nums):
            # 计算目标差值
            complement = target - num
            # 如果差值已经在哈希表中，说明找到了结果
            if complement in seen:
                return [seen[complement], i]
            # 否则将当前元素存入哈希表
            seen[num] = i
        return []  # 如果没有找到符合条件的答案

```
