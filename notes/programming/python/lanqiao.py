"""
C++刷刷刷题题题 - Python 实现版本
所有题目的Python解答
"""

# ==================== 1. 迷宫 (动态规划) ====================
def maze_problem():
    """
    迷宫问题 - 计算从左上到右下的最小成本
    """
    n, m, k, p = map(int, input().split())
    
    # 读取初始成本
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row = list(map(int, input().split()))
        for j in range(1, m + 1):
            cost[i][j] = row[j - 1]
    
    # 读取障碍物
    for _ in range(k):
        x, y, z = map(int, input().split())
        cost[x][y] += z
    
    # 读取陷阱
    trap = [['' for _ in range(m + 1)] for _ in range(n + 1)]
    for _ in range(p):
        parts = input().split()
        x, y, c = int(parts[0]), int(parts[1]), parts[2]
        trap[x][y] = c
    
    # DP初始化
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dp[1][1] = cost[1][1]
    
    # 动态规划
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if i == 1 and j == 1:
                continue
            
            # 常规移动
            if i > 1:
                dp[i][j] = min(dp[i][j], dp[i-1][j] + cost[i][j])
            if j > 1:
                dp[i][j] = min(dp[i][j], dp[i][j-1] + cost[i][j])
            
            # 陷阱向下移动两步
            if i > 2 and trap[i-1][j] == 'D':
                dp[i][j] = min(dp[i][j], dp[i-2][j] + cost[i-1][j] + cost[i][j])
            
            # 陷阱向右移动两步
            if j > 2 and trap[i][j-1] == 'R':
                dp[i][j] = min(dp[i][j], dp[i][j-2] + cost[i][j-1] + cost[i][j])
    
    print(dp[n][m])


# ==================== 2. 不同角度 (字符串字典序) ====================
def different_angle():
    """
    不同角度 - 找到字典序大于S的最小T
    """
    t = int(input())
    results = []
    
    for _ in range(t):
        s = int(input())
        if s == 0:
            results.append(1)
        else:
            results.append(s * 10)
    
    for r in results:
        print(r)


# ==================== 3. 铠甲合体 (贪心+哈希) ====================
def armor_combination():
    """
    铠甲合体 - 贪心选择能量值组合
    """
    n, m = map(int, input().split())
    
    # 读取能量值并统计
    energy_list = list(map(int, input().split()))
    from collections import Counter
    energy_count = Counter(energy_list)
    
    # 降序排列
    sorted_energies = sorted(energy_count.items(), reverse=True)
    
    # 读取战斗力
    battle_powers = list(map(int, input().split()))
    
    results = []
    for power in battle_powers:
        count = 0
        remaining = power
        
        # 贪心选择
        for energy, available in sorted_energies:
            use = min(available, remaining // energy)
            count += use
            remaining -= use * energy
        
        if remaining > 0:
            results.append(-1)
        else:
            results.append(count)
    
    print(' '.join(map(str, results)))


# ==================== 4. 连连看 (哈希表枚举) ====================
def lian_lian_kan_v1():
    """
    连连看 - 解法1: 哈希表枚举
    """
    n, m = map(int, input().split())
    
    # 哈希表存储元素及其坐标
    from collections import defaultdict
    positions = defaultdict(list)
    
    for i in range(n):
        row = list(map(int, input().split()))
        for j in range(m):
            positions[row[j]].append((i, j))
    
    count = 0
    for value, coords in positions.items():
        # 检查所有配对
        for i in range(len(coords)):
            for j in range(len(coords)):
                a, b = coords[i]
                c, d = coords[j]
                if abs(a - c) == abs(b - d) and abs(a - c) > 0:
                    count += 1
    
    print(count)


def lian_lian_kan_v2():
    """
    连连看 - 解法2: 对角线检查
    """
    n, m = map(int, input().split())
    
    arr = []
    for i in range(n):
        arr.append(list(map(int, input().split())))
    
    count = 0
    for i in range(n):
        for j in range(m):
            # 检查右下方向
            a, b = i, j
            while True:
                a += 1
                b += 1
                if a >= n or b >= m:
                    break
                if arr[i][j] == arr[a][b]:
                    count += 2
            
            # 检查右上方向
            a, b = i, j
            while True:
                a += 1
                b -= 1
                if a >= n or b < 0:
                    break
                if arr[i][j] == arr[a][b]:
                    count += 2
    
    print(count)


def lian_lian_kan_optimized():
    """
    连连看 - 最优 Python 实现
    复杂度 O(n*m)，利用对角线哈希表计算
    """
    n, m = map(int, input().split())
    arr = [list(map(int, input().split())) for _ in range(n)]

    # 哈希表记录每个 value 出现在哪些对角线
    from collections import defaultdict
    main_diag = defaultdict(lambda: defaultdict(int))   # 主对角线 a - b
    anti_diag = defaultdict(lambda: defaultdict(int))   # 副对角线 a + b

    for i in range(n):
        for j in range(m):
            val = arr[i][j]
            main_diag[val][i - j] += 1
            anti_diag[val][i + j] += 1

    count = 0
    # 统计组合数 k*(k-1) → 每对匹配贡献 2
    for val in main_diag:
        for k in main_diag[val].values():
            if k > 1:
                count += k * (k - 1)
    for val in anti_diag:
        for k in anti_diag[val].values():
            if k > 1:
                count += k * (k - 1)

    print(count)


# ==================== 5. 星际旅行 (BFS图遍历) ====================
def star_travel():
    """
    星际旅行 - BFS计算可达星球数量的期望
    """
    from collections import deque
    
    n, m, Q = map(int, input().split())
    
    # 建图
    graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    counts = []
    
    for _ in range(Q):
        x, y = map(int, input().split())
        
        # BFS
        visited = [False] * (n + 1)
        queue = deque([(x, 0)])
        visited[x] = True
        can_reach = 0
        
        while queue:
            current, steps = queue.popleft()
            can_reach += 1
            
            if steps < y:
                for neighbor in graph[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, steps + 1))
        
        counts.append(can_reach)
    
    # 计算期望
    expectation = sum(counts) / Q
    print(f"{expectation:.2f}")


# ==================== 6. 购物车里的宝贝 (位运算) ====================
def shopping_cart():
    """
    购物车里的宝贝 - 异或和判断
    """
    n = int(input())
    prices = list(map(int, input().split()))
    
    total_xor = 0
    for price in prices:
        total_xor ^= price
    
    if total_xor == 0:
        print("YES")
    else:
        print("NO")


# ==================== 7. 01背包问题 (动态规划) ====================
def knapsack_01():
    """
    01背包问题 - 每个物品只能选一次
    """
    total_weight, N = map(int, input().split())
    
    weights = []
    values = []
    for _ in range(N):
        w, v = map(int, input().split())
        weights.append(w)
        values.append(v)
    
    # DP数组
    dp = [[0] * (total_weight + 1) for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(total_weight + 1):
            if weights[i - 1] <= j:
                dp[i][j] = max(dp[i - 1][j], 
                             values[i - 1] + dp[i - 1][j - weights[i - 1]])
            else:
                dp[i][j] = dp[i - 1][j]
    
    print(dp[N][total_weight])


# ==================== 8. 完全背包问题 (动态规划) ====================
def knapsack_complete():
    """
    完全背包问题 - 每个物品可以无限选用
    """
    total_weight, N = map(int, input().split())
    
    weights = []
    values = []
    for _ in range(N):
        w, v = map(int, input().split())
        weights.append(w)
        values.append(v)
    
    # DP数组
    dp = [0] * (total_weight + 1)
    
    for i in range(N):
        for j in range(weights[i], total_weight + 1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    
    print(dp[total_weight])


# ==================== 9. 图书管理员 (字符串后缀匹配) ====================
def library_manager():
    """
    图书管理员 - 后缀匹配查找最小编码
    """
    n, q = map(int, input().split())
    
    book_codes = []
    for _ in range(n):
        book_codes.append(int(input()))
    
    book_codes.sort()  # 升序排列
    
    results = []
    for _ in range(q):
        length, req = map(int, input().split())
        
        found = False
        for code in book_codes:
            # 检查code是否以req结尾
            if code % (10 ** length) == req:
                results.append(code)
                found = True
                break
        
        if not found:
            results.append(-1)
    
    for r in results:
        print(r)


# ==================== 10. 健身 (完全背包+分段) ====================
def fitness():
    """
    健身 - 完全背包问题的分段应用
    """
    n, m, q = map(int, input().split())
    
    # 读取不能健身的日期
    unavailable = list(map(int, input().split()))
    unavailable.sort()
    
    # 读取健身计划
    train_days = []
    incomes = []
    for _ in range(m):
        k, s = map(int, input().split())
        train_days.append(2 ** k)
        incomes.append(s)
    
    # 计算每个时间段
    periods = []
    periods.append(unavailable[0] - 1)
    for i in range(1, q):
        periods.append(unavailable[i] - unavailable[i - 1] - 1)
    periods.append(n - unavailable[-1])
    
    # 对每个时间段做完全背包
    def complete_knapsack(capacity, weights, values):
        dp = [0] * (capacity + 1)
        for i in range(len(weights)):
            for j in range(weights[i], capacity + 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        return dp[capacity]
    
    total_income = 0
    for period in periods:
        if period > 0:
            total_income += complete_knapsack(period, train_days, incomes)
    
    print(total_income)


# ==================== 11. Hanoi双塔问题 (递归/递推) ====================
def hanoi_double_tower_recursive():
    """
    Hanoi双塔 - 递归解法
    """
    def hanoi_moves(n):
        if n == 1:
            return 2
        return 2 * hanoi_moves(n - 1) + 2
    
    n = int(input())
    print(hanoi_moves(n))


def hanoi_double_tower_iterative():
    """
    Hanoi双塔 - 递推高精度解法
    """
    def add_big_numbers(a, b):
        result = []
        carry = 0
        i, j = len(a) - 1, len(b) - 1
        
        while i >= 0 or j >= 0 or carry:
            total = carry
            if i >= 0:
                total += int(a[i])
                i -= 1
            if j >= 0:
                total += int(b[j])
                j -= 1
            carry = total // 10
            result.append(str(total % 10))
        
        return ''.join(reversed(result))
    
    n = int(input())
    result = "2"
    
    for i in range(2, n + 1):
        result = add_big_numbers(add_big_numbers(result, result), "2")
    
    print(result)


# ==================== 12. 包子凑数 (完全背包+GCD) ====================
def baozi_count():
    """
    包子凑数 - 判断能凑出的数目
    """
    import math
    
    N = int(input())
    A = []
    
    for _ in range(N):
        A.append(int(input()))
    
    # 计算最大公约数
    g = A[0]
    for i in range(1, N):
        g = math.gcd(g, A[i])
    
    # 如果最大公约数不是1，输出INF
    if g != 1:
        print("INF")
        return
    
    # 完全背包求能凑出的数
    MAX_SUM = 10001
    dp = [False] * MAX_SUM
    dp[0] = True
    
    for i in range(N):
        for j in range(A[i], MAX_SUM):
            if dp[j - A[i]]:
                dp[j] = True
    
    # 统计不能凑出的数
    count = sum(1 for i in range(1, MAX_SUM) if not dp[i])
    print(count)


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("C++刷题集 - Python实现")
    print("=" * 60)
    print("\n请选择要运行的题目:")
    print("1. 迷宫")
    print("2. 不同角度")
    print("3. 铠甲合体")
    print("4. 连连看 (解法1)")
    print("5. 连连看 (解法2)")
    print("6. 连连看 (最优解法)")
    print("7. 星际旅行")
    print("8. 购物车里的宝贝")
    print("9. 01背包问题")
    print("10. 完全背包问题")
    print("11. 图书管理员")
    print("12. 健身")
    print("13. Hanoi双塔 (递归)")
    print("14. Hanoi双塔 (递推)")
    print("15. 包子凑数")

    choice = input("\n请输入题号 (1-15): ").strip()
    
    problems = {
        "1": maze_problem,
        "2": different_angle,
        "3": armor_combination,
        "4": lian_lian_kan_v1,
        "5": lian_lian_kan_v2,
        "6": lian_lian_kan_optimized,
        "7": star_travel,
        "8": shopping_cart,
        "9": knapsack_01,
        "10": knapsack_complete,
        "11": library_manager,
        "12": fitness,
        "13": hanoi_double_tower_recursive,
        "14": hanoi_double_tower_iterative,
        "15": baozi_count
    }
    
    if choice in problems:
        print(f"\n{'=' * 60}")
        print(f"运行题目 {choice}")
        print(f"{'=' * 60}\n")
        problems[choice]()
    else:
        print("无效的题号！")