# 强化学习完整笔记

## 目录
1. [基础概念](#一基础概念)
2. [马尔可夫决策过程](#二马尔可夫决策过程mdp)
3. [动态规划](#三动态规划)
4. [蒙特卡洛方法](#四蒙特卡洛方法)
5. [时序差分学习](#五时序差分学习)
6. [值函数逼近](#六值函数逼近)
7. [策略梯度方法](#七策略梯度方法)
8. [Actor-Critic](#八actor-critic方法)
9. [深度强化学习](#九深度强化学习)
10. [高级算法](#十高级算法)

---

## 一、基础概念

### 1.1 强化学习简介
```
强化学习（Reinforcement Learning, RL）是机器学习的一个分支，
研究智能体（Agent）如何在环境（Environment）中通过试错学习，
以最大化累积奖励。

核心特点：
- 延迟奖励：行动的结果可能在很久之后才显现
- 探索与利用：需要平衡尝试新策略和利用已知好策略
- 序贯决策：当前决策影响未来状态
```

### 1.2 基本要素
```python
# 强化学习的五元组
class RLSystem:
    def __init__(self):
        # 状态空间 (State Space)
        self.states = S
        
        # 动作空间 (Action Space)
        self.actions = A
        
        # 奖励函数 (Reward Function)
        # R: S × A → ℝ
        self.reward_function = R
        
        # 状态转移概率 (Transition Probability)
        # P: S × A × S → [0,1]
        self.transition_prob = P
        
        # 折扣因子 (Discount Factor)
        # γ ∈ [0,1]
        self.gamma = 0.99

# 智能体与环境交互
def agent_environment_interaction():
    """
    智能体-环境交互循环：
    1. 智能体观察状态 s_t
    2. 智能体选择动作 a_t
    3. 环境返回奖励 r_t 和新状态 s_{t+1}
    4. 重复
    """
    state = env.reset()
    
    for t in range(max_steps):
        # 智能体选择动作
        action = agent.select_action(state)
        
        # 环境响应
        next_state, reward, done, info = env.step(action)
        
        # 智能体学习
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            break
```

### 1.3 回报与价值函数
```python
import numpy as np

# 1. 回报 (Return)
def compute_return(rewards, gamma=0.99):
    """
    计算累积折扣回报
    G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... = Σ γ^k * r_{t+k}
    """
    G = 0
    for t in range(len(rewards)-1, -1, -1):
        G = rewards[t] + gamma * G
    return G

# 示例
rewards = [1, 0, 0, 1, 1]
print(f"回报: {compute_return(rewards)}")  # 2.9701

# 2. 状态价值函数 (State Value Function)
def state_value_function():
    """
    V^π(s) = 𝔼_π[G_t | s_t = s]
           = 𝔼_π[r_t + γV^π(s_{t+1}) | s_t = s]
    
    表示在状态s下，遵循策略π能获得的期望回报
    """
    pass

# 3. 动作价值函数 (Action Value Function)
def action_value_function():
    """
    Q^π(s,a) = 𝔼_π[G_t | s_t = s, a_t = a]
             = 𝔼[r_t + γQ^π(s_{t+1}, a_{t+1}) | s_t = s, a_t = a]
    
    表示在状态s下采取动作a，然后遵循策略π的期望回报
    """
    pass

# 4. 优势函数 (Advantage Function)
def advantage_function(Q, V, state, action):
    """
    A^π(s,a) = Q^π(s,a) - V^π(s)
    
    表示在状态s下采取动作a相比平均水平的优势
    """
    return Q[state, action] - V[state]
```

### 1.4 策略
```python
# 1. 确定性策略 (Deterministic Policy)
class DeterministicPolicy:
    """
    π: S → A
    每个状态映射到唯一的动作
    """
    def __init__(self, policy_dict):
        self.policy = policy_dict
    
    def select_action(self, state):
        return self.policy[state]

# 2. 随机策略 (Stochastic Policy)
class StochasticPolicy:
    """
    π: S × A → [0,1]
    π(a|s) 表示在状态s下选择动作a的概率
    """
    def __init__(self, policy_probs):
        self.policy = policy_probs
    
    def select_action(self, state):
        actions = list(self.policy[state].keys())
        probs = list(self.policy[state].values())
        return np.random.choice(actions, p=probs)

# 3. ε-贪心策略 (ε-greedy Policy)
class EpsilonGreedyPolicy:
    """
    以概率ε随机探索，以概率1-ε选择最优动作
    """
    def __init__(self, Q, epsilon=0.1):
        self.Q = Q
        self.epsilon = epsilon
    
    def select_action(self, state, actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(actions)  # 探索
        else:
            return np.argmax(self.Q[state])   # 利用

# 4. Softmax策略 (Boltzmann策略)
class SoftmaxPolicy:
    """
    π(a|s) = exp(Q(s,a)/τ) / Σ_a' exp(Q(s,a')/τ)
    τ是温度参数，控制探索程度
    """
    def __init__(self, Q, temperature=1.0):
        self.Q = Q
        self.tau = temperature
    
    def select_action(self, state, actions):
        q_values = self.Q[state]
        probs = np.exp(q_values / self.tau)
        probs = probs / np.sum(probs)
        return np.random.choice(actions, p=probs)
```

### 1.5 探索策略
```python
# 1. ε衰减
class EpsilonDecay:
    """随着训练进行，减少探索"""
    def __init__(self, epsilon_start=1.0, epsilon_end=0.01, decay_steps=10000):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
    
    def get_epsilon(self, step):
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                  min(step / self.decay_steps, 1.0)
        return epsilon

# 2. 上置信界 (UCB - Upper Confidence Bound)
class UCBExploration:
    """
    选择 a = argmax_a [Q(s,a) + c * sqrt(ln(t) / N(s,a))]
    平衡利用（Q值）和探索（访问次数少的动作）
    """
    def __init__(self, c=2.0):
        self.c = c
        self.N = {}  # 访问计数
        self.t = 0   # 总步数
    
    def select_action(self, state, Q, actions):
        self.t += 1
        if state not in self.N:
            self.N[state] = {a: 0 for a in actions}
        
        ucb_values = []
        for a in actions:
            if self.N[state][a] == 0:
                return a  # 优先选择未访问的动作
            
            ucb = Q[state][a] + self.c * np.sqrt(np.log(self.t) / self.N[state][a])
            ucb_values.append(ucb)
        
        action = actions[np.argmax(ucb_values)]
        self.N[state][action] += 1
        return action

# 3. Thompson采样
class ThompsonSampling:
    """
    基于贝叶斯推断的探索策略
    为每个动作维护一个分布，从分布中采样
    """
    def __init__(self):
        self.alpha = {}  # 成功计数
        self.beta = {}   # 失败计数
    
    def select_action(self, state, actions):
        if state not in self.alpha:
            self.alpha[state] = {a: 1 for a in actions}
            self.beta[state] = {a: 1 for a in actions}
        
        samples = {}
        for a in actions:
            # 从Beta分布采样
            samples[a] = np.random.beta(self.alpha[state][a], self.beta[state][a])
        
        return max(samples, key=samples.get)
    
    def update(self, state, action, reward):
        if reward > 0:
            self.alpha[state][action] += 1
        else:
            self.beta[state][action] += 1
```

---

## 二、马尔可夫决策过程(MDP)

### 2.1 MDP定义
```python
class MDP:
    """
    马尔可夫决策过程五元组: (S, A, P, R, γ)
    
    - S: 状态空间
    - A: 动作空间
    - P: 状态转移概率 P(s'|s,a)
    - R: 奖励函数 R(s,a,s')
    - γ: 折扣因子
    """
    def __init__(self, states, actions, transitions, rewards, gamma=0.99):
        self.states = states
        self.actions = actions
        self.P = transitions  # P[s][a][s'] = 概率
        self.R = rewards      # R[s][a][s'] = 奖励
        self.gamma = gamma
    
    def get_transition_prob(self, state, action, next_state):
        """获取状态转移概率"""
        return self.P[state][action].get(next_state, 0)
    
    def get_reward(self, state, action, next_state):
        """获取奖励"""
        return self.R[state][action].get(next_state, 0)

# 示例：网格世界MDP
class GridWorldMDP:
    """
    简单的4×4网格世界
    目标：从起点(0,0)到达终点(3,3)
    """
    def __init__(self):
        self.grid_size = 4
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = 0.9
        
    def get_next_state(self, state, action):
        """确定性状态转移"""
        x, y = state
        
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.grid_size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.grid_size - 1, y + 1)
        
        return (x, y)
    
    def get_reward(self, state, action, next_state):
        """奖励函数"""
        if next_state == (3, 3):  # 终点
            return 1.0
        return -0.01  # 每步小惩罚，鼓励快速到达终点
    
    def is_terminal(self, state):
        """判断是否为终止状态"""
        return state == (3, 3)
```

### 2.2 贝尔曼方程
```python
# 1. 贝尔曼期望方程 (Bellman Expectation Equation)
def bellman_expectation_v(V, mdp, policy, state):
    """
    状态价值函数的贝尔曼期望方程
    V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
    """
    value = 0
    for action in mdp.actions:
        action_prob = policy.get_action_prob(state, action)
        
        for next_state in mdp.states:
            trans_prob = mdp.get_transition_prob(state, action, next_state)
            reward = mdp.get_reward(state, action, next_state)
            
            value += action_prob * trans_prob * (reward + mdp.gamma * V[next_state])
    
    return value

def bellman_expectation_q(Q, mdp, policy, state, action):
    """
    动作价值函数的贝尔曼期望方程
    Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s')Q^π(s',a')]
    """
    value = 0
    for next_state in mdp.states:
        trans_prob = mdp.get_transition_prob(state, action, next_state)
        reward = mdp.get_reward(state, action, next_state)
        
        next_value = 0
        for next_action in mdp.actions:
            action_prob = policy.get_action_prob(next_state, next_action)
            next_value += action_prob * Q[next_state, next_action]
        
        value += trans_prob * (reward + mdp.gamma * next_value)
    
    return value

# 2. 贝尔曼最优方程 (Bellman Optimality Equation)
def bellman_optimality_v(V, mdp, state):
    """
    最优状态价值函数
    V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
    """
    max_value = float('-inf')
    
    for action in mdp.actions:
        value = 0
        for next_state in mdp.states:
            trans_prob = mdp.get_transition_prob(state, action, next_state)
            reward = mdp.get_reward(state, action, next_state)
            value += trans_prob * (reward + mdp.gamma * V[next_state])
        
        max_value = max(max_value, value)
    
    return max_value

def bellman_optimality_q(Q, mdp, state, action):
    """
    最优动作价值函数
    Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]
    """
    value = 0
    
    for next_state in mdp.states:
        trans_prob = mdp.get_transition_prob(state, action, next_state)
        reward = mdp.get_reward(state, action, next_state)
        max_next_q = max(Q[next_state, a] for a in mdp.actions)
        value += trans_prob * (reward + mdp.gamma * max_next_q)
    
    return value
```

### 2.3 马尔可夫性质
```python
class MarkovProperty:
    """
    马尔可夫性质：未来只依赖于当前状态，与历史无关
    P(s_{t+1} | s_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t)
    """
    
    @staticmethod
    def check_markov_property(trajectory, transition_counts):
        """
        检查轨迹是否满足马尔可夫性质
        通过比较 P(s'|s) 和 P(s'|s,history)
        """
        # 计算单步转移概率
        single_step = {}
        for i in range(len(trajectory) - 1):
            s, s_next = trajectory[i], trajectory[i+1]
            if s not in single_step:
                single_step[s] = {}
            single_step[s][s_next] = single_step[s].get(s_next, 0) + 1
        
        # 归一化
        for s in single_step:
            total = sum(single_step[s].values())
            for s_next in single_step[s]:
                single_step[s][s_next] /= total
        
        # 计算带历史的转移概率
        history_step = {}
        for i in range(2, len(trajectory)):
            history = tuple(trajectory[:i])
            s_next = trajectory[i]
            if history not in history_step:
                history_step[history] = {}
            history_step[history][s_next] = history_step[history].get(s_next, 0) + 1
        
        # 比较差异
        # （实际应用中需要更严格的统计测试）
        return single_step, history_step

# 部分可观测马尔可夫决策过程 (POMDP)
class POMDP:
    """
    当智能体无法完全观测状态时，使用POMDP
    
    七元组: (S, A, P, R, Ω, O, γ)
    - Ω: 观测空间
    - O: 观测概率 O(o|s,a)
    """
    def __init__(self, states, actions, observations, 
                 transitions, rewards, obs_probs, gamma=0.99):
        self.states = states
        self.actions = actions
        self.observations = observations
        self.P = transitions
        self.R = rewards
        self.O = obs_probs  # O[s][a][o] = 概率
        self.gamma = gamma
        
        # 信念状态 (Belief State)
        self.belief = self.initialize_belief()
    
    def initialize_belief(self):
        """初始化均匀信念"""
        n = len(self.states)
        return {s: 1.0/n for s in self.states}
    
    def update_belief(self, belief, action, observation):
        """
        贝叶斯信念更新
        b'(s') ∝ O(o|s',a) Σ_s P(s'|s,a)b(s)
        """
        new_belief = {}
        
        for s_next in self.states:
            prob = 0
            for s in self.states:
                prob += self.P[s][action].get(s_next, 0) * belief[s]
            prob *= self.O[s_next][action].get(observation, 0)
            new_belief[s_next] = prob
        
        # 归一化
        total = sum(new_belief.values())
        if total > 0:
            new_belief = {s: p/total for s, p in new_belief.items()}
        
        return new_belief
```

---

## 三、动态规划

### 3.1 策略评估
```python
import numpy as np

class PolicyEvaluation:
    """
    策略评估：计算给定策略π的价值函数V^π
    使用迭代方法求解贝尔曼期望方程
    """
    def __init__(self, mdp, policy, theta=1e-6):
        self.mdp = mdp
        self.policy = policy
        self.theta = theta  # 收敛阈值
    
    def evaluate(self, max_iterations=1000):
        """
        迭代策略评估
        V_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV_k(s')]
        """
        # 初始化价值函数
        V = {s: 0 for s in self.mdp.states}
        
        for iteration in range(max_iterations):
            delta = 0
            new_V = V.copy()
            
            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    continue
                
                v = V[state]
                
                # 贝尔曼期望更新
                new_value = 0
                for action in self.mdp.actions:
                    action_prob = self.policy.get_action_prob(state, action)
                    
                    for next_state in self.mdp.states:
                        trans_prob = self.mdp.get_transition_prob(state, action, next_state)
                        reward = self.mdp.get_reward(state, action, next_state)
                        new_value += action_prob * trans_prob * \
                                   (reward + self.mdp.gamma * V[next_state])
                
                new_V[state] = new_value
                delta = max(delta, abs(v - new_value))
            
            V = new_V
            
            # 检查收敛
            if delta < self.theta:
                print(f"策略评估收敛于第 {iteration+1} 次迭代")
                break
        
        return V

# 示例：网格世界策略评估
def example_policy_evaluation():
    # 创建4×4网格世界
    mdp = GridWorldMDP()
    
    # 定义随机策略（每个方向概率相等）
    policy = UniformPolicy(mdp.actions)
    
    # 评估策略
    evaluator = PolicyEvaluation(mdp, policy)
    V = evaluator.evaluate()
    
    # 打印价值函数
    print("状态价值函数:")
    for i in range(4):
        for j in range(4):
            print(f"{V[(i,j)]:.2f}", end="  ")
        print()
```

### 3.2 策略改进
```python
class PolicyImprovement:
    """
    策略改进：根据价值函数改进策略
    π'(s) = argmax_a Q^π(s,a)
    """
    def __init__(self, mdp):
        self.mdp = mdp
    
    def improve(self, V):
        """
        贪心策略改进
        """
        new_policy = {}
        policy_stable = True
        
        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                continue
            
            # 计算每个动作的Q值
            q_values = {}
            for action in self.mdp.actions:
                q = 0
                for next_state in self.mdp.states:
                    trans_prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    q += trans_prob * (reward + self.mdp.gamma * V[next_state])
                q_values[action] = q
            
            policy[state] = max(q_values, key=q_values.get)
        
        return policy

# 异步动态规划
class AsynchronousDP:
    """
    异步动态规划：不需要完整扫描所有状态
    - 原地更新 (In-place)
    - 优先级扫描
    - 实时动态规划
    """
    def __init__(self, mdp):
        self.mdp = mdp
    
    def prioritized_sweeping(self, V, theta=1e-6, max_iterations=1000):
        """
        优先级扫描：优先更新Bellman误差大的状态
        """
        import heapq
        
        # 初始化优先队列（最大堆，用负值实现）
        priority_queue = []
        
        # 计算初始优先级
        for state in self.mdp.states:
            if not self.mdp.is_terminal(state):
                bellman_error = self.compute_bellman_error(state, V)
                if bellman_error > theta:
                    heapq.heappush(priority_queue, (-bellman_error, state))
        
        for iteration in range(max_iterations):
            if not priority_queue:
                break
            
            # 取出误差最大的状态
            _, state = heapq.heappop(priority_queue)
            
            # 更新该状态
            old_value = V[state]
            V[state] = self.bellman_backup(state, V)
            
            # 更新前驱状态的优先级
            for prev_state in self.get_predecessors(state):
                if not self.mdp.is_terminal(prev_state):
                    bellman_error = self.compute_bellman_error(prev_state, V)
                    if bellman_error > theta:
                        heapq.heappush(priority_queue, (-bellman_error, prev_state))
        
        return V
    
    def bellman_backup(self, state, V):
        """执行Bellman更新"""
        max_value = float('-inf')
        
        for action in self.mdp.actions:
            value = 0
            for next_state in self.mdp.states:
                trans_prob = self.mdp.get_transition_prob(state, action, next_state)
                reward = self.mdp.get_reward(state, action, next_state)
                value += trans_prob * (reward + self.mdp.gamma * V[next_state])
            max_value = max(max_value, value)
        
        return max_value
    
    def compute_bellman_error(self, state, V):
        """计算Bellman误差"""
        new_value = self.bellman_backup(state, V)
        return abs(V[state] - new_value)
    
    def get_predecessors(self, state):
        """获取可以转移到该状态的前驱状态"""
        predecessors = []
        for s in self.mdp.states:
            for a in self.mdp.actions:
                if self.mdp.get_transition_prob(s, a, state) > 0:
                    predecessors.append(s)
        return predecessors
```

---

## 四、蒙特卡洛方法

### 4.1 蒙特卡洛预测
```python
class MonteCarloPredictor:
    """
    蒙特卡洛策略评估
    通过采样完整轨迹估计价值函数
    不需要环境模型（model-free）
    """
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.returns = {}  # 记录每个状态的回报
        self.V = {}        # 状态价值函数
    
    def first_visit_mc(self, episodes):
        """
        首次访问MC：只统计状态第一次出现时的回报
        """
        for episode in episodes:
            # episode = [(s0,a0,r0), (s1,a1,r1), ..., (sT,aT,rT)]
            states_visited = set()
            G = 0
            
            # 从后向前计算回报
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                # 首次访问
                if state not in states_visited:
                    states_visited.add(state)
                    
                    if state not in self.returns:
                        self.returns[state] = []
                    self.returns[state].append(G)
                    
                    # 更新价值函数（平均）
                    self.V[state] = np.mean(self.returns[state])
        
        return self.V
    
    def every_visit_mc(self, episodes):
        """
        每次访问MC：统计状态所有出现时的回报
        """
        for episode in episodes:
            G = 0
            
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                if state not in self.returns:
                    self.returns[state] = []
                self.returns[state].append(G)
                self.V[state] = np.mean(self.returns[state])
        
        return self.V
    
    def incremental_mc(self, episodes, alpha=None):
        """
        增量式MC：使用增量平均更新
        V(s) ← V(s) + α[G - V(s)]
        """
        for state in self.V:
            if state not in self.returns:
                self.returns[state] = []
        
        for episode in episodes:
            G = 0
            states_visited = set()
            
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                if state not in states_visited:
                    states_visited.add(state)
                    
                    if state not in self.V:
                        self.V[state] = 0
                    
                    # 增量更新
                    if alpha is None:
                        # 自适应学习率
                        n = len(self.returns.get(state, [])) + 1
                        step_size = 1.0 / n
                    else:
                        step_size = alpha
                    
                    self.V[state] += step_size * (G - self.V[state])
        
        return self.V

# 重要性采样
class ImportanceSamplingMC:
    """
    重要性采样：使用行为策略采样，评估目标策略
    适用于离策略（off-policy）学习
    """
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.V = {}
        self.C = {}  # 累积权重
    
    def weighted_importance_sampling(self, episodes, behavior_policy, target_policy):
        """
        加权重要性采样
        ρ_t = π(a_t|s_t) / b(a_t|s_t)
        """
        for episode in episodes:
            G = 0
            W = 1.0  # 重要性采样比率
            
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                if state not in self.C:
                    self.C[state] = 0
                    self.V[state] = 0
                
                self.C[state] += W
                
                # 加权更新
                self.V[state] += (W / self.C[state]) * (G - self.V[state])
                
                # 更新重要性采样比率
                pi_prob = target_policy.get_action_prob(state, action)
                b_prob = behavior_policy.get_action_prob(state, action)
                
                if b_prob == 0:
                    break
                
                W *= pi_prob / b_prob
                
                if W == 0:
                    break
        
        return self.V
```

### 4.2 蒙特卡洛控制
```python
class MonteCarloControl:
    """
    蒙特卡洛控制：通过采样学习最优策略
    """
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}  # Q(s,a)
        self.returns = {}
        self.policy = {}
    
    def on_policy_mc_control(self, num_episodes=1000):
        """
        同策略MC控制（ε-贪心策略改进）
        1. 使用ε-贪心策略生成轨迹
        2. 评估Q函数
        3. 改进策略
        """
        for episode_num in range(num_episodes):
            # 生成轨迹
            episode = self.generate_episode()
            
            # 更新Q值（首次访问）
            states_actions_visited = set()
            G = 0
            
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                sa_pair = (state, action)
                if sa_pair not in states_actions_visited:
                    states_actions_visited.add(sa_pair)
                    
                    if sa_pair not in self.returns:
                        self.returns[sa_pair] = []
                    self.returns[sa_pair].append(G)
                    
                    # 更新Q值
                    if state not in self.Q:
                        self.Q[state] = {}
                    self.Q[state][action] = np.mean(self.returns[sa_pair])
                    
                    # 策略改进（ε-贪心）
                    self.update_epsilon_greedy_policy(state)
        
        return self.Q, self.policy
    
    def off_policy_mc_control(self, num_episodes=1000):
        """
        离策略MC控制
        行为策略：ε-贪心（用于探索）
        目标策略：贪心（要学习的策略）
        """
        # 初始化目标策略为贪心
        target_policy = {}
        
        # 累积权重
        C = {}
        
        for episode_num in range(num_episodes):
            # 使用行为策略生成轨迹
            episode = self.generate_episode()
            
            G = 0
            W = 1.0
            
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                sa_pair = (state, action)
                
                if sa_pair not in C:
                    C[sa_pair] = 0
                    if state not in self.Q:
                        self.Q[state] = {}
                    if action not in self.Q[state]:
                        self.Q[state][action] = 0
                
                C[sa_pair] += W
                
                # 加权更新Q值
                self.Q[state][action] += (W / C[sa_pair]) * \
                                        (G - self.Q[state][action])
                
                # 更新目标策略（贪心）
                if state in self.Q and len(self.Q[state]) > 0:
                    target_policy[state] = max(self.Q[state], 
                                              key=self.Q[state].get)
                
                # 如果动作不是贪心动作，终止
                if state not in target_policy or action != target_policy[state]:
                    break
                
                # 更新重要性采样比率
                # 行为策略是ε-贪心，目标策略是贪心
                num_actions = len(self.env.action_space)
                b_prob = self.epsilon / num_actions + \
                        (1 - self.epsilon) * (action == target_policy.get(state))
                pi_prob = 1.0  # 贪心策略
                
                W *= pi_prob / b_prob
        
        return self.Q, target_policy
    
    def generate_episode(self):
        """使用当前策略生成一个轨迹"""
        episode = []
        state = self.env.reset()
        done = False
        
        while not done:
            # ε-贪心选择动作
            action = self.select_epsilon_greedy_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def select_epsilon_greedy_action(self, state):
        """ε-贪心动作选择"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if state not in self.Q or len(self.Q[state]) == 0:
                return self.env.action_space.sample()
            return max(self.Q[state], key=self.Q[state].get)
    
    def update_epsilon_greedy_policy(self, state):
        """更新ε-贪心策略"""
        if state in self.Q and len(self.Q[state]) > 0:
            self.policy[state] = max(self.Q[state], key=self.Q[state].get)
```

---

## 五、时序差分学习

### 5.1 TD预测
```python
class TDPredictor:
    """
    时序差分（TD）预测
    结合了MC和DP的优点：
    - 像MC一样是model-free的
    - 像DP一样可以bootstrap（用估计更新估计）
    """
    def __init__(self, gamma=0.99, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.V = {}
    
    def td_0(self, env, policy, num_episodes=1000):
        """
        TD(0)：单步时序差分
        V(s_t) ← V(s_t) + α[r_t + γV(s_{t+1}) - V(s_t)]
        
        TD目标：r_t + γV(s_{t+1})
        TD误差：δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            while not done:
                # 根据策略选择动作
                action = policy.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # 初始化V值
                if state not in self.V:
                    self.V[state] = 0
                if next_state not in self.V:
                    self.V[next_state] = 0
                
                # TD更新
                td_target = reward + self.gamma * self.V[next_state] * (not done)
                td_error = td_target - self.V[state]
                self.V[state] += self.alpha * td_error
                
                state = next_state
        
        return self.V
    
    def td_lambda(self, env, policy, lambda_=0.9, num_episodes=1000):
        """
        TD(λ)：使用资格迹
        结合了多步TD的优点
        
        λ = 0: TD(0)
        λ = 1: MC
        """
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            # 资格迹
            eligibility_trace = {}
            
            while not done:
                action = policy.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                if state not in self.V:
                    self.V[state] = 0
                if next_state not in self.V:
                    self.V[next_state] = 0
                
                # TD误差
                td_error = reward + self.gamma * self.V[next_state] * (not done) - self.V[state]
                
                # 更新资格迹
                if state not in eligibility_trace:
                    eligibility_trace[state] = 0
                eligibility_trace[state] += 1
                
                # 更新所有状态的价值（根据资格迹）
                for s in eligibility_trace:
                    self.V[s] += self.alpha * td_error * eligibility_trace[s]
                    eligibility_trace[s] *= self.gamma * lambda_
                
                state = next_state
        
        return self.V
    
    def n_step_td(self, env, policy, n=5, num_episodes=1000):
        """
        n步TD
        G_t^{(n)} = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^nV(s_{t+n})
        """
        for episode in range(num_episodes):
            # 存储轨迹
            states = [env.reset()]
            actions = []
            rewards = [0]  # 填充，使索引对齐
            
            T = float('inf')
            t = 0
            
            while True:
                if t < T:
                    action = policy.select_action(states[t])
                    next_state, reward, done, _ = env.step(action)
                    
                    states.append(next_state)
                    actions.append(action)
                    rewards.append(reward)
                    
                    if done:
                        T = t + 1
                
                # 更新时刻
                tau = t - n + 1
                
                if tau >= 0:
                    # 计算n步回报
                    G = sum([self.gamma**(i-tau-1) * rewards[i] 
                            for i in range(tau+1, min(tau+n, T)+1)])
                    
                    if tau + n < T:
                        state_tau_n = states[tau + n]
                        if state_tau_n not in self.V:
                            self.V[state_tau_n] = 0
                        G += self.gamma**n * self.V[state_tau_n]
                    
                    # 更新
                    state_tau = states[tau]
                    if state_tau not in self.V:
                        self.V[state_tau] = 0
                    self.V[state_tau] += self.alpha * (G - self.V[state_tau])
                
                if tau == T - 1:
                    break
                
                t += 1
        
        return self.V
```

### 5.2 SARSA
```python
class SARSA:
    """
    SARSA：同策略TD控制算法
    State-Action-Reward-State-Action
    
    Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    """
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = {}
    
    def train(self, num_episodes=1000):
        """SARSA训练"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.select_action(state)
            
            total_reward = 0
            done = False
            
            while not done:
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                # 选择下一个动作（同策略）
                next_action = self.select_action(next_state)
                
                # SARSA更新
                self.update_q(state, action, reward, next_state, next_action, done)
                
                state = next_state
                action = next_action
            
            episode_rewards.append(total_reward)
            
            # ε衰减
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return self.Q, episode_rewards
    
    def update_q(self, state, action, reward, next_state, next_action, done):
        """SARSA Q值更新"""
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = 0
        
        if not done:
            if next_state not in self.Q:
                self.Q[next_state] = {}
            if next_action not in self.Q[next_state]:
                self.Q[next_state][next_action] = 0
            
            td_target = reward + self.gamma * self.Q[next_state][next_action]
        else:
            td_target = reward
        
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
    
    def select_action(self, state):
        """ε-贪心动作选择"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if state not in self.Q or len(self.Q[state]) == 0:
                return self.env.action_space.sample()
            return max(self.Q[state], key=self.Q[state].get)

# SARSA(λ)：带资格迹的SARSA
class SARSALambda(SARSA):
    """
    SARSA(λ)：结合资格迹的SARSA
    可以更快地传播奖励信号
    """
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1, lambda_=0.9):
        super().__init__(env, gamma, alpha, epsilon)
        self.lambda_ = lambda_
    
    def train(self, num_episodes=1000):
        """SARSA(λ)训练"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.select_action(state)
            
            # 资格迹
            eligibility_trace = {}
            
            total_reward = 0
            done = False
            
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                next_action = self.select_action(next_state)
                
                # 初始化Q值
                if state not in self.Q:
                    self.Q[state] = {}
                if action not in self.Q[state]:
                    self.Q[state][action] = 0
                
                # TD误差
                if not done:
                    if next_state not in self.Q:
                        self.Q[next_state] = {}
                    if next_action not in self.Q[next_state]:
                        self.Q[next_state][next_action] = 0
                    td_error = reward + self.gamma * self.Q[next_state][next_action] - \
                              self.Q[state][action]
                else:
                    td_error = reward - self.Q[state][action]
                
                # 更新资格迹
                if state not in eligibility_trace:
                    eligibility_trace[state] = {}
                if action not in eligibility_trace[state]:
                    eligibility_trace[state][action] = 0
                eligibility_trace[state][action] += 1
                
                # 更新所有状态-动作对
                for s in list(eligibility_trace.keys()):
                    for a in list(eligibility_trace[s].keys()):
                        if s not in self.Q:
                            self.Q[s] = {}
                        if a not in self.Q[s]:
                            self.Q[s][a] = 0
                        
                        self.Q[s][a] += self.alpha * td_error * eligibility_trace[s][a]
                        eligibility_trace[s][a] *= self.gamma * self.lambda_
                        
                        # 清除很小的资格迹
                        if eligibility_trace[s][a] < 1e-5:
                            del eligibility_trace[s][a]
                
                state = next_state
                action = next_action
            
            episode_rewards.append(total_reward)
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return self.Q, episode_rewards
```

### 5.3 Q-Learning
```python
class QLearning:
    """
    Q-Learning：离策略TD控制算法
    
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    关键特点：
    - 行为策略：ε-贪心（用于探索）
    - 目标策略：贪心（max_a'）
    """
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = {}
    
    def train(self, num_episodes=1000):
        """Q-Learning训练"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # 使用ε-贪心选择动作（行为策略）
                action = self.select_action(state)
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                # Q-Learning更新
                self.update_q(state, action, reward, next_state, done)
                
                state = next_state
            
            episode_rewards.append(total_reward)
            
            # ε衰减
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return self.Q, episode_rewards
    
    def update_q(self, state, action, reward, next_state, done):
        """Q-Learning更新规则"""
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = 0
        
        if not done:
            # 使用max（贪心，目标策略）
            if next_state not in self.Q or len(self.Q[next_state]) == 0:
                max_next_q = 0
            else:
                max_next_q = max(self.Q[next_state].values())
            
            td_target = reward + self.gamma * max_next_q
        else:
            td_target = reward
        
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
    
    def select_action(self, state):
        """ε-贪心动作选择"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if state not in self.Q or len(self.Q[state]) == 0:
                return self.env.action_space.sample()
            return max(self.Q[state], key=self.Q[state].get)
    
    def get_greedy_action(self, state):
        """获取贪心动作（用于测试）"""
        if state not in self.Q or len(self.Q[state]) == 0:
            return self.env.action_space.sample()
        return max(self.Q[state], key=self.Q[state].get)

# Double Q-Learning
class DoubleQLearning:
    """
    Double Q-Learning：解决Q-Learning的过估计问题
    维护两个Q函数：Q1和Q2
    
    更新Q1时，用Q1选择动作，用Q2评估
    更新Q2时，用Q2选择动作，用Q1评估
    """
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q1 = {}
        self.Q2 = {}
    
    def train(self, num_episodes=1000):
        """Double Q-Learning训练"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                # 随机选择更新Q1或Q2
                if np.random.random() < 0.5:
                    self.update_q1(state, action, reward, next_state, done)
                else:
                    self.update_q2(state, action, reward, next_state, done)
                
                state = next_state
            
            episode_rewards.append(total_reward)
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return self.Q1, self.Q2, episode_rewards
    
    def update_q1(self, state, action, reward, next_state, done):
        """使用Q2评估Q1选择的动作"""
        if state not in self.Q1:
            self.Q1[state] = {}
        if action not in self.Q1[state]:
            self.Q1[state][action] = 0
        
        if not done:
            # Q1选择动作
            if next_state not in self.Q1 or len(self.Q1[next_state]) == 0:
                best_action = self.env.action_space.sample()
            else:
                best_action = max(self.Q1[next_state], key=self.Q1[next_state].get)
            
            # Q2评估
            if next_state not in self.Q2 or best_action not in self.Q2[next_state]:
                next_q = 0
            else:
                next_q = self.Q2[next_state][best_action]
            
            td_target = reward + self.gamma * next_q
        else:
            td_target = reward
        
        self.Q1[state][action] += self.alpha * (td_target - self.Q1[state][action])
    
    def update_q2(self, state, action, reward, next_state, done):
        """使用Q1评估Q2选择的动作"""
        if state not in self.Q2:
            self.Q2[state] = {}
        if action not in self.Q2[state]:
            self.Q2[state][action] = 0
        
        if not done:
            # Q2选择动作
            if next_state not in self.Q2 or len(self.Q2[next_state]) == 0:
                best_action = self.env.action_space.sample()
            else:
                best_action = max(self.Q2[next_state], key=self.Q2[next_state].get)
            
            # Q1评估
            if next_state not in self.Q1 or best_action not in self.Q1[next_state]:
                next_q = 0
            else:
                next_q = self.Q1[next_state][best_action]
            
            td_target = reward + self.gamma * next_q
        else:
            td_target = reward
        
        self.Q2[state][action] += self.alpha * (td_target - self.Q2[state][action])
    
    def select_action(self, state):
        """使用Q1+Q2的平均值进行ε-贪心"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # 合并Q1和Q2
        q_avg = {}
        all_actions = set()
        
        if state in self.Q1:
            all_actions.update(self.Q1[state].keys())
        if state in self.Q2:
            all_actions.update(self.Q2[state].keys())
        
        if not all_actions:
            return self.env.action_space.sample()
        
        for a in all_actions:
            q1_val = self.Q1.get(state, {}).get(a, 0)
            q2_val = self.Q2.get(state, {}).get(a, 0)
            q_avg[a] = (q1_val + q2_val) / 2
        
        return max(q_avg, key=q_avg.get)
```

---

## 六、值函数逼近

### 6.1 线性函数逼近
```python
import numpy as np

class LinearFunctionApproximation:
    """
    线性函数逼近
    V(s) ≈ φ(s)ᵀw
    Q(s,a) ≈ φ(s,a)ᵀw
    
    其中φ是特征向量，w是权重
    """
    def __init__(self, feature_dim, alpha=0.01):
        self.w = np.zeros(feature_dim)
        self.alpha = alpha
    
    def predict_value(self, features):
        """预测状态价值"""
        return np.dot(features, self.w)
    
    def update(self, features, target):
        """
        梯度下降更新
        w ← w + α[target - V(s)]∇V(s)
        w ← w + α[target - V(s)]φ(s)
        """
        prediction = self.predict_value(features)
        error = target - prediction
        self.w += self.alpha * error * features
    
    def semi_gradient_td(self, env, feature_extractor, num_episodes=1000, gamma=0.99):
        """
        半梯度TD(0)
        只对价值函数的梯度进行更新，不对目标的梯度更新
        """
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            while not done:
                # 提取特征
                features = feature_extractor(state)
                
                # 选择动作（这里假设有策略）
                action = env.action_space.sample()
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                
                # TD更新
                if not done:
                    next_features = feature_extractor(next_state)
                    td_target = reward + gamma * self.predict_value(next_features)
                else:
                    td_target = reward
                
                self.update(features, td_target)
                state = next_state
        
        return self.w

# 特征工程示例
class FeatureExtractor:
    """特征提取器"""
    
    @staticmethod
    def polynomial_features(state, degree=2):
        """多项式特征"""
        state = np.array(state).flatten()
        features = [1]  # 偏置项
        
        # 一阶特征
        features.extend(state)
        
        # 高阶特征
        if degree >= 2:
            for i in range(len(state)):
                for j in range(i, len(state)):
                    features.append(state[i] * state[j])
        
        return np.array(features)
    
    @staticmethod
    def tile_coding(state, num_tilings=8, num_tiles=8):
        """
        Tile Coding：将连续状态空间分成多个重叠的网格
        每个网格称为一个tiling
        """
        features = []
        state = np.array(state).flatten()
        
        for tiling in range(num_tilings):
            # 为每个tiling添加偏移
            offset = tiling / num_tilings
            
            for dim in range(len(state)):
                # 计算该维度的tile索引
                tile_idx = int((state[dim] + offset) * num_tiles)
                tile_idx = max(0, min(num_tiles - 1, tile_idx))
                
                # 创建one-hot编码
                one_hot = np.zeros(num_tiles)
                one_hot[tile_idx] = 1
                features.extend(one_hot)
        
        return np.array(features)
    
    @staticmethod
    def rbf_features(state, centers, sigma=1.0):
        """
        径向基函数（RBF）特征
        φ_i(s) = exp(-||s - c_i||² / (2σ²))
        """
        state = np.array(state).flatten()
        features = []
        
        for center in centers:
            center = np.array(center).flatten()
            distance = np.linalg.norm(state - center)
            feature = np.exp(-distance**2 / (2 * sigma**2))
            features.append(feature)
        
        return np.array(features)

# 线性Q函数逼近
class LinearQApproximation:
    """
    线性Q函数逼近
    Q(s,a) = φ(s,a)ᵀw
    """
    def __init__(self, num_actions, feature_dim, alpha=0.01):
        self.num_actions = num_actions
        # 为每个动作维护一个权重向量
        self.w = np.zeros((num_actions, feature_dim))
        self.alpha = alpha
    
    def predict_q(self, features, action):
        """预测Q(s,a)"""
        return np.dot(features, self.w[action])
    
    def predict_all_q(self, features):
        """预测所有动作的Q值"""
        return features @ self.w.T
    
    def update_q_learning(self, features, action, reward, next_features, done, gamma=0.99):
        """
        Q-Learning更新
        w ← w + α[r + γ max_a' Q(s',a') - Q(s,a)]φ(s,a)
        """
        current_q = self.predict_q(features, action)
        
        if not done:
            max_next_q = np.max(self.predict_all_q(next_features))
            td_target = reward + gamma * max_next_q
        else:
            td_target = reward
        
        td_error = td_target - current_q
        self.w[action] += self.alpha * td_error * features
    
    def update_sarsa(self, features, action, reward, next_features, next_action, done, gamma=0.99):
        """
        SARSA更新
        w ← w + α[r + γQ(s',a') - Q(s,a)]φ(s,a)
        """
        current_q = self.predict_q(features, action)
        
        if not done:
            next_q = self.predict_q(next_features, next_action)
            td_target = reward + gamma * next_q
        else:
            td_target = reward
        
        td_error = td_target - current_q
        self.w[action] += self.alpha * td_error * features
```

### 6.2 神经网络函数逼近
```python
import torch
import torch.nn as nn
import torch.optim as optim

class ValueNetwork(nn.Module):
    """
    使用神经网络逼近价值函数
    """
    def __init__(self, state_dim, hidden_dims=[64, 64]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))  # 输出单个值
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

class QNetwork(nn.Module):
    """
    使用神经网络逼近Q函数
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))  # 输出所有动作的Q值
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

# 使用神经网络的TD学习
class NeuralTD:
    """使用神经网络的TD学习"""
    def __init__(self, state_dim, lr=0.001, gamma=0.99):
        self.value_net = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
    
    def train_step(self, state, reward, next_state, done):
        """单步训练"""
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        # 前向传播
        current_value = self.value_net(state_tensor)
        
        # 计算TD目标
        with torch.no_grad():
            if not done:
                next_value = self.value_net(next_state_tensor)
                td_target = reward + self.gamma * next_value
            else:
                td_target = torch.FloatTensor([reward])
        
        # 计算损失
        loss = nn.MSELoss()(current_value, td_target)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Dueling Network
class DuelingQNetwork(nn.Module):
    """
    Dueling网络架构
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    
    分离价值和优势，学习更稳定
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DuelingQNetwork, self).__init__()
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 结合价值和优势
        # Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values
```

---

## 七、策略梯度方法

### 7.1 REINFORCE算法
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """策略网络：输出动作概率分布"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class REINFORCE:
    """
    REINFORCE算法（蒙特卡洛策略梯度）
    
    ∇J(θ) = 𝔼_π[∇log π(a|s,θ) * G_t]
    
    基本思想：
    - 好的动作（高回报）→增加概率
    - 坏的动作（低回报）→降低概率
    """
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        """根据策略选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def train(self, env, num_episodes=1000):
        """训练REINFORCE"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            # 收集一个完整轨迹
            states, actions, rewards, log_probs = [], [], [], []
            
            state = env.reset()
            done = False
            
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                state = next_state
            
            # 计算回报
            returns = self.compute_returns(rewards)
            
            # 策略梯度更新
            policy_loss = []
            for log_prob, G in zip(log_probs, returns):
                policy_loss.append(-log_prob * G)
            
            policy_loss = torch.stack(policy_loss).sum()
            
            # 反向传播
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            episode_rewards.append(sum(rewards))
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
        
        return episode_rewards
    
    def compute_returns(self, rewards):
        """计算折扣回报"""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # 标准化回报（减少方差）
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns

# 带基线的REINFORCE
class REINFORCEWithBaseline:
    """
    带基线的REINFORCE
    使用价值函数作为基线减少方差
    
    ∇J(θ) = 𝔼[∇log π(a|s,θ) * (G_t - b(s))]
    其中b(s)是基线，通常用V(s)
    """
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def train(self, env, num_episodes=1000):
        episode_rewards = []
        
        for episode in range(num_episodes):
            states, actions, rewards, log_probs, values = [], [], [], [], []
            
            state = env.reset()
            done = False
            
            while not done:
                action, log_prob = self.select_action(state)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                value = self.value_net(state_tensor)
                
                next_state, reward, done, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                
                state = next_state
            
            # 计算回报和优势
            returns = self.compute_returns(rewards)
            values = torch.cat(values).squeeze()
            advantages = returns - values.detach()
            
            # 更新策略网络
            policy_loss = []
            for log_prob, advantage in zip(log_probs, advantages):
                policy_loss.append(-log_prob * advantage)
            policy_loss = torch.stack(policy_loss).sum()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 更新价值网络
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            episode_rewards.append(sum(rewards))
        
        return episode_rewards
    
    def compute_returns(self, rewards):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)
```

### 7.2 Actor-Critic基础
```python
class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络
    Actor：输出策略π(a|s)
    Critic：输出价值V(s)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic头
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value

class ActorCritic:
    """
    Actor-Critic算法（单步）
    优势：在线学习，低方差
    """
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.ac_net = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.ac_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
    
    def train(self, env, num_episodes=1000):
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # 选择动作
                action, log_prob, value = self.select_action(state)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                # 计算TD误差（优势）
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                _, next_value = self.ac_net(next_state_tensor)
                
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * next_value.item()
                
                advantage = td_target - value.item()
                
                # 计算损失
                actor_loss = -log_prob * advantage
                critic_loss = F.mse_loss(value, torch.FloatTensor([td_target]))
                
                loss = actor_loss + critic_loss
                
                # 更新网络
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                state = next_state
            
            episode_rewards.append(total_reward)
        
        return episode_rewards
```

---

## 八、Actor-Critic方法

### 8.1 A2C (Advantage Actor-Critic)
```python
class A2C:
    """
    Advantage Actor-Critic (A2C)
    使用优势函数：A(s,a) = Q(s,a) - V(s)
    """
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, 
                 value_coef=0.5, entropy_coef=0.01):
        self.ac_net = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """
        批量更新
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 前向传播
        action_probs, values = self.ac_net(states)
        _, next_values = self.ac_net(next_states)
        
        # 计算优势
        td_targets = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = td_targets - values.squeeze()
        
        # Actor损失（策略梯度）
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic损失（TD误差）
        critic_loss = advantages.pow(2).mean()
        
        # 熵正则化（鼓励探索）
        entropy = dist.entropy().mean()
        
        # 总损失
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()

### 8.2 A3C (Asynchronous Advantage Actor-Critic)
```python
import torch.multiprocessing as mp

class A3C:
    """
    A3C：异步优势Actor-Critic
    使用多个并行worker收集经验
    """
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        # 全局网络（共享参数）
        self.global_net = ActorCriticNetwork(state_dim, action_dim)
        self.global_net.share_memory()
        
        self.optimizer = optim.Adam(self.global_net.parameters(), lr=lr)
        self.gamma = gamma
    
    def worker(self, worker_id, env_fn, num_episodes):
        """
        Worker进程：独立采样和计算梯度
        """
        # 创建本地网络
        local_net = ActorCriticNetwork(state_dim, action_dim)
        env = env_fn()
        
        for episode in range(num_episodes):
            # 同步全局参数到本地
            local_net.load_state_dict(self.global_net.state_dict())
            
            # 收集轨迹
            states, actions, rewards = [], [], []
            state = env.reset()
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, _ = local_net(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample()
                
                next_state, reward, done, _ = env.step(action.item())
                
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                
                state = next_state
            
            # 计算损失并更新全局网络
            self.update_global(local_net, states, actions, rewards)
    
    def update_global(self, local_net, states, actions, rewards):
        """更新全局网络"""
        # 计算回报
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # 计算损失
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        
        action_probs, values = local_net(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        advantages = returns - values.squeeze().detach()
        
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        entropy = dist.entropy().mean()
        
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # 计算梯度
        self.optimizer.zero_grad()
        loss.backward()
        
        # 将本地梯度应用到全0
                for next_state in self.mdp.states:
                    trans_prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    q += trans_prob * (reward + self.mdp.gamma * V[next_state])
                q_values[action] = q
            
            # 选择最优动作
            best_action = max(q_values, key=q_values.get)
            new_policy[state] = best_action
            
            # 检查策略是否改变
            if state in new_policy and new_policy[state] != best_action:
                policy_stable = False
        
        return new_policy, policy_stable
```

### 3.3 策略迭代
```python
class PolicyIteration:
    """
    策略迭代：交替进行策略评估和策略改进
    1. 策略评估：计算V^π
    2. 策略改进：π' = greedy(V^π)
    3. 重复直到策略收敛
    """
    def __init__(self, mdp, theta=1e-6):
        self.mdp = mdp
        self.theta = theta
    
    def iterate(self, max_iterations=100):
        """策略迭代主循环"""
        # 初始化随机策略
        policy = {s: np.random.choice(self.mdp.actions) 
                 for s in self.mdp.states}
        V = {s: 0 for s in self.mdp.states}
        
        for iteration in range(max_iterations):
            # 1. 策略评估
            V = self.policy_evaluation(policy, V)
            
            # 2. 策略改进
            new_policy, policy_stable = self.policy_improvement(V)
            
            if policy_stable:
                print(f"策略迭代收敛于第 {iteration+1} 次迭代")
                break
            
            policy = new_policy
        
        return policy, V
    
    def policy_evaluation(self, policy, V):
        """策略评估子过程"""
        while True:
            delta = 0
            new_V = V.copy()
            
            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    continue
                
                action = policy[state]
                new_value = 0
                
                for next_state in self.mdp.states:
                    trans_prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    new_value += trans_prob * (reward + self.mdp.gamma * V[next_state])
                
                new_V[state] = new_value
                delta = max(delta, abs(V[state] - new_value))
            
            V = new_V
            
            if delta < self.theta:
                break
        
        return V
    
    def policy_improvement(self, V):
        """策略改进子过程"""
        new_policy = {}
        policy_stable = True
        
        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                continue
            
            # 计算Q值
            q_values = {}
            for action in self.mdp.actions:
                q = 0
                for next_state in self.mdp.states:
                    trans_prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    q += trans_prob * (reward + self.mdp.gamma * V[next_state])
                q_values[action] = q
            
            best_action = max(q_values, key=q_values.get)
            new_policy[state] = best_action
        
        return new_policy, policy_stable
```

### 3.4 价值迭代
```python
class ValueIteration:
    """
    价值迭代：直接迭代最优贝尔曼方程
    V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV_k(s')]
    
    相比策略迭代，价值迭代更简单高效
    """
    def __init__(self, mdp, theta=1e-6):
        self.mdp = mdp
        self.theta = theta
    
    def iterate(self, max_iterations=1000):
        """价值迭代主循环"""
        # 初始化价值函数
        V = {s: 0 for s in self.mdp.states}
        
        for iteration in range(max_iterations):
            delta = 0
            new_V = V.copy()
            
            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    continue
                
                # 贝尔曼最优更新
                max_value = float('-inf')
                
                for action in self.mdp.actions:
                    value = 0
                    for next_state in self.mdp.states:
                        trans_prob = self.mdp.get_transition_prob(state, action, next_state)
                        reward = self.mdp.get_reward(state, action, next_state)
                        value += trans_prob * (reward + self.mdp.gamma * V[next_state])
                    
                    max_value = max(max_value, value)
                
                new_V[state] = max_value
                delta = max(delta, abs(V[state] - new_value))
            
            V = new_V
            
            if delta < self.theta:
                print(f"价值迭代收敛于第 {iteration+1} 次迭代")
                break
        
        # 提取最优策略
        policy = self.extract_policy(V)
        
        return policy, V
    
    def extract_policy(self, V):
        """从价值函数提取策略"""
        policy = {}
        
        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                continue
            
            q_values = {}
            for action in self.mdp.actions:
                q = 0
                for next_state in self.mdp.states:
                    trans_prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    q += trans_prob * (reward + self.mdp.gamma * V[next_state])
                q_values[action] = q
            
            policy[state] = max(q_values, key=q_values.get)
        
        return policy
```

---

## 九、深度强化学习

### 9.1 DQN (Deep Q-Network)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

# 经验回放缓冲区
Transition = namedtuple('Transition', 
                       ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样批量经验"""
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        
        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.FloatTensor(batch.done)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    """DQN网络架构"""
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

class DQN:
    """
    Deep Q-Network
    
    关键技术：
    1. 经验回放 (Experience Replay)
    2. 目标网络 (Target Network)
    3. ε-贪心探索
    """
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10):
        
        # Q网络和目标网络
        self.q_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 超参数
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.action_dim = action_dim
        self.update_counter = 0
    
    def select_action(self, state, training=True):
        """ε-贪心动作选择"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state)
            return q_values.argmax().item()
    
    def train_step(self):
        """SAC训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 更新Critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新Actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新温度参数
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # 软更新目标网络
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()
    
    def soft_update(self, source, target):
        """软更新目标网络"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class GaussianPolicy(nn.Module):
    """高斯策略网络"""
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
    
    def forward(self, state):
        x = self.network(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 重参数化技巧
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # 计算log概率
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        action = action * self.max_action
        mean = torch.tanh(mean) * self.max_action
        
        return action, log_prob, mean
```

### 10.4 TD3 (Twin Delayed DDPG)

```python
class TD3:
    """
    TD3：双延迟DDPG
    
    改进：
    1. 双Critic网络（减少过估计）
    2. 延迟策略更新
    3. 目标策略平滑
    """
    def __init__(self, state_dim, action_dim, max_action,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        
        # Actor
        self.actor = DDPGActor(state_dim, action_dim, max_action)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 双Critic
        self.critic1 = DDPGCritic(state_dim, action_dim)
        self.critic2 = DDPGCritic(state_dim, action_dim)
        self.critic1_target = DDPGCritic(state_dim, action_dim)
        self.critic2_target = DDPGCritic(state_dim, action_dim)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        self.replay_buffer = ReplayBuffer(100000)
        self.batch_size = 256
        self.total_iterations = 0
    
    def select_action(self, state, noise=0.1):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def train_step(self):
        """TD3训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        self.total_iterations += 1
        
        # 采样
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 更新Critic
        with torch.no_grad():
            # 目标策略平滑
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # 计算目标Q值（取最小值）
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))
        
        # 更新两个Critic
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 延迟策略更新
        if self.total_iterations % self.policy_delay == 0:
            # 更新Actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
            
            return actor_loss.item(), critic1_loss.item()
        
        return None, critic1_loss.item()
    
    def soft_update(self, source, target):
        """软更新"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

---

## 十一、多智能体强化学习

### 11.1 独立学习

```python
class IndependentQLearning:
    """
    独立Q学习：每个智能体独立学习
    将其他智能体视为环境的一部分
    """
    def __init__(self, num_agents, state_dim, action_dim, lr=0.1, gamma=0.99, epsilon=0.1):
        self.num_agents = num_agents
        self.agents = [
            QLearning(None, gamma, lr, epsilon) 
            for _ in range(num_agents)
        ]
    
    def select_actions(self, states):
        """所有智能体选择动作"""
        return [agent.select_action(state) for agent, state in zip(self.agents, states)]
    
    def update(self, states, actions, rewards, next_states, dones):
        """更新所有智能体"""
        for i, agent in enumerate(self.agents):
            agent.update_q(states[i], actions[i], rewards[i], next_states[i], dones[i])
```

### 11.2 MADDPG (Multi-Agent DDPG)

```python
class MADDPG:
    """
    MADDPG：多智能体DDPG
    
    关键思想：
    - 集中式训练（Critic看到全局信息）
    - 分布式执行（Actor只用局部信息）
    """
    def __init__(self, num_agents, state_dims, action_dims, 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.01):
        
        self.num_agents = num_agents
        self.agents = []
        
        # 全局状态和动作维度
        total_state_dim = sum(state_dims)
        total_action_dim = sum(action_dims)
        
        for i in range(num_agents):
            agent = {
                'actor': DDPGActor(state_dims[i], action_dims[i], 1.0),
                'actor_target': DDPGActor(state_dims[i], action_dims[i], 1.0),
                'critic': DDPGCritic(total_state_dim, total_action_dim),
                'critic_target': DDPGCritic(total_state_dim, total_action_dim),
                'actor_optimizer': None,
                'critic_optimizer': None
            }
            
            agent['actor_target'].load_state_dict(agent['actor'].state_dict())
            agent['critic_target'].load_state_dict(agent['critic'].state_dict())
            
            agent['actor_optimizer'] = optim.Adam(agent['actor'].parameters(), lr=actor_lr)
            agent['critic_optimizer'] = optim.Adam(agent['critic'].parameters(), lr=critic_lr)
            
            self.agents.append(agent)
        
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(100000)
        self.batch_size = 1024
    
    def select_actions(self, states, noise=0.0):
        """所有智能体选择动作"""
        actions = []
        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.agents[i]['actor'](state_tensor).detach().numpy()[0]
            
            if noise > 0:
                action += np.random.normal(0, noise, size=action.shape)
            
            actions.append(action)
        
        return actions
    
    def train_step(self):
        """MADDPG训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样（假设buffer存储了全局信息）
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # states: [batch, num_agents, state_dim]
        # actions: [batch, num_agents, action_dim]
        
        # 展平全局状态和动作
        batch_size = states.shape[0]
        global_states = states.reshape(batch_size, -1)
        global_actions = actions.reshape(batch_size, -1)
        global_next_states = next_states.reshape(batch_size, -1)
        
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            
            # 更新Critic（使用全局信息）
            with torch.no_grad():
                # 获取所有智能体的下一个动作
                next_actions_list = []
                for i in range(self.num_agents):
                    next_action = self.agents[i]['actor_target'](next_states[:, i])
                    next_actions_list.append(next_action)
                
                global_next_actions = torch.cat(next_actions_list, dim=-1)
                
                target_q = agent['critic_target'](global_next_states, global_next_actions)
                target_q = rewards[:, agent_id].unsqueeze(1) + \
                          self.gamma * target_q * (1 - dones[:, agent_id].unsqueeze(1))
            
            current_q = agent['critic'](global_states, global_actions)
            critic_loss = F.mse_loss(current_q, target_q)
            
            agent['critic_optimizer'].zero_grad()
            critic_loss.backward()
            agent['critic_optimizer'].step()
            
            # 更新Actor（只用局部状态）
            # 构造当前智能体的动作，其他智能体动作来自当前策略
            actions_list = []
            for i in range(self.num_agents):
                if i == agent_id:
                    action = agent['actor'](states[:, i])
                else:
                    action = self.agents[i]['actor'](states[:, i]).detach()
                actions_list.append(action)
            
            global_actions_for_actor = torch.cat(actions_list, dim=-1)
            
            actor_loss = -agent['critic'](global_states, global_actions_for_actor).mean()
            
            agent['actor_optimizer'].zero_grad()
            actor_loss.backward()
            agent['actor_optimizer'].step()
            
            # 软更新
            self.soft_update(agent['actor'], agent['actor_target'])
            self.soft_update(agent['critic'], agent['critic_target'])
    
    def soft_update(self, source, target):
        """软更新"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

---

## 十二、模型学习

### 12.1 Dyna-Q

```python
class DynaQ:
    """
    Dyna-Q：结合模型学习和无模型学习
    
    流程：
    1. 与环境交互（真实经验）
    2. 更新Q值
    3. 更新环境模型
    4. 用模型生成模拟经验
    5. 用模拟经验更新Q值
    """
    def __init__(self, state_dim, action_dim, lr=0.1, gamma=0.99, 
                 epsilon=0.1, planning_steps=5):
        
        self.Q = {}
        self.model = {}  # model[s][a] = (r, s')
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        
        self.action_dim = action_dim
        self.visited_states = set()
    
    def select_action(self, state):
        """ε-贪心选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        if state not in self.Q or len(self.Q[state]) == 0:
            return random.randrange(self.action_dim)
        
        return max(self.Q[state], key=self.Q[state].get)
    
    def update(self, state, action, reward, next_state, done):
        """Dyna-Q更新"""
        # 1. 直接RL更新（真实经验）
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = 0
        
        if not done:
            if next_state not in self.Q or len(self.Q[next_state]) == 0:
                max_next_q = 0
            else:
                max_next_q = max(self.Q[next_state].values())
            td_target = reward + self.gamma * max_next_q
        else:
            td_target = reward
        
        self.Q[state][action] += self.lr * (td_target - self.Q[state][action])
        
        # 2. 更新模型
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (reward, next_state, done)
        self.visited_states.add(state)
        
        # 3. 规划（使用模型生成模拟经验）
        for _ in range(self.planning_steps):
            # 随机选择访问过的状态
            s = random.choice(list(self.visited_states))
            
            # 随机选择该状态下执行过的动作
            if s not in self.model or len(self.model[s]) == 0:
                continue
            
            a = random.choice(list(self.model[s].keys()))
            
            # 从模型获取转移
            r, s_next, d = self.model[s][a]
            
            # 更新Q值（模拟经验）
            if s not in self.Q:
                self.Q[s] = {}
            if a not in self.Q[s]:
                self.Q[s][a] = 0
            
            if not d:
                if s_next not in self.Q or len(self.Q[s_next]) == 0:
                    max_next_q = 0
                else:
                    max_next_q = max(self.Q[s_next].values())
                td_target = r + self.gamma * max_next_q
            else:
                td_target = r
            
            self.Q[s][a] += self.lr * (td_target - self.Q[s][a])
```

### 12.2 世界模型

```python
class WorldModel(nn.Module):
    """
    世界模型：学习环境动态
    预测下一个状态和奖励
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(WorldModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 预测下一个状态
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        
        # 预测奖励
        self.reward_head = nn.Linear(hidden_dim, 1)
        
        # 预测终止
        self.done_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        """
        预测：(next_state, reward, done)
        """
        x = torch.cat([state, action], dim=-1)
        features = self.network(x)
        
        next_state = self.next_state_head(features)
        reward = self.reward_head(features)
        done = torch.sigmoid(self.done_head(features))
        
        return next_state, reward, done

class MBPO:
    """
    Model-Based Policy Optimization
    结合模型学习和策略优化
    """
    def __init__(self, state_dim, action_dim, max_action):
        # 世界模型
        self.world_model = WorldModel(state_dim, action_dim)
        self.model_optimizer = optim.Adam(self.world_model.parameters(), lr=1e-3)
        
        # 策略（使用SAC）
        self.policy = SAC(state_dim, action_dim, max_action)
        
        # 真实和模拟经验缓冲区
        self.real_buffer = ReplayBuffer(100000)
        self.model_buffer = ReplayBuffer(100000)
    
    def train_world_model(self, num_epochs=5):
        """训练世界模型"""
        for _ in range(num_epochs):
            if len(self.real_buffer) < 256:
                continue
            
            states, actions, rewards, next_states, dones = \
                self.real_buffer.sample(256)
            
            # 预测
            pred_next_states, pred_rewards, pred_dones = \
                self.world_model(states, actions)
            
            # 计算损失
            state_loss = F.mse_loss(pred_next_states, next_states)
            reward_loss = F.mse_loss(pred_rewards.squeeze(), rewards)
            done_loss = F.binary_cross_entropy(pred_dones.squeeze(), dones)
            
            loss = state_loss + reward_loss + done_loss
            
            # 更新
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
    
    def generate_model_data(self, num_samples=10000):
        """使用世界模型生成模拟数据"""
        if len(self.real_buffer) < 256:
            return
        
        # 从真实数据采样初始状态
        states, _, _, _, _ = self.real_buffer.sample(num_samples)
        
        for state in states:
            # 使用当前策略选择动作
            action = self.policy.select_action(state.numpy())
            
            # 使用世界模型预测转移
            with torch.no_grad():
                state_tensor = state.unsqueeze(0)
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                
                next_state, reward, done = self.world_model(state_tensor, action_tensor)
                
                next_state = next_state.squeeze().numpy()
                reward = reward.item()
                done = (done.item() > 0.5)
            
            # 存储到模型缓冲区
            self.model_buffer.push(state.numpy(), action, reward, next_state, done)
```

---

## 十三、逆强化学习

### 13.1 最大熵IRL

```python
class MaxEntIRL:
    """
    最大熵逆强化学习
    从专家演示中学习奖励函数
    """
    def __init__(self, state_dim, action_dim, lr=0.01):
        # 奖励函数参数化
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=lr)
    
    def compute_reward(self, state):
        """计算奖励"""
        state = torch.FloatTensor(state)
        return self.reward_net(state).item()
    
    def train(self, expert_trajectories, policy, num_iterations=100):
        """
        训练IRL
        
        目标：最大化专家轨迹的似然
        同时最小化策略轨迹与专家轨迹的差异
        """
        for iteration in range(num_iterations):
            # 1. 计算专家轨迹的特征期望
            expert_features = self.compute_feature_expectations(expert_trajectories)
            
            # 2. 用当前奖励训练策略
            policy.train_with_reward(self.reward_net, num_episodes=10)
            
            # 3. 生成策略轨迹
            policy_trajectories = policy.generate_trajectories(num_episodes=10)
            
            # 4. 计算策略轨迹的特征期望
            policy_features = self.compute_feature_expectations(policy_trajectories)
            
            # 5. 更新奖励函数
            loss = -torch.sum(expert_features * torch.log(policy_features + 1e-8))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def compute_feature_expectations(self, trajectories):
        """计算特征期望"""
        features = []
        for traj in trajectories:
            for state, action, _ in traj:
                state_tensor = torch.FloatTensor(state)
                feature = self.reward_net(state_tensor)
                features.append(feature)
        
        return torch.stack(features).mean(dim=0)
```

---

## 十四、实用技巧与调试

### 14.1 超参数调优

```python
class HyperparameterTuning:
    """超参数调优建议"""
    
    @staticmethod
    def learning_rate_schedule():
        """学习率调度"""
        # 1. 线性衰减
        def linear_decay(initial_lr, final_lr, total_steps):
            def schedule(step):
                fraction = min(step / total_steps, 1.0)
                return initial_lr - (initial_lr - final_lr) * fraction
            return schedule
        
        # 2. 余弦退火
        def cosine_annealing(initial_lr, final_lr, total_steps):
            def schedule(step):
                fraction = step / total_steps
                return final_lr + 0.5 * (initial_lr - final_lr) * \
                       (1 + np.cos(np.pi * fraction))
            return schedule
        
        return linear_decay, cosine_annealing
    
    @staticmethod
    def exploration_schedule():
        """探索策略调度"""
        # ε-贪心衰减
        def epsilon_decay(start=1.0, end=0.01, decay_steps=10000):
            def schedule(step):
                return max(end, start - (start - end) * min(step / decay_steps, 1.0))
            return schedule
        
        return epsilon_decay
    
    @staticmethod
    def common_ranges():
        """常用超参数范围"""
        return {
            'learning_rate': [1e-5, 1e-4, 3e-4, 1e-3, 3e-3],
            'gamma': [0.95, 0.99, 0.995, 0.999],
            'batch_size': [32, 64, 128, 256, 512],
            'hidden_dims': [[64, 64], [128, 128], [256, 256]],
            'buffer_size': [10000, 50000, 100000, 1000000],
            'tau': [0.001, 0.005, 0.01, 0.05]
        }
```

### 14.2 调试技巧

```python
class DebuggingTools:
    """强化学习调试工具"""
    
    @staticmethod
    def plot_learning_curve(rewards, window=100):
        """绘制学习曲线"""
        import matplotlib.pyplot as plt
        
        # 平滑奖励
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, alpha=0.3, label='Raw')
        plt.plot(smoothed, label=f'{window}-episode Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.title('Learning Curve')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def analyze_q_values(Q, states):
        """分析Q值分布"""
        import matplotlib.pyplot as plt
        
        q_values = []
        for state in states:
            if state in Q:
                q_values.extend(Q[state].values())
        
        plt.figure(figsize=(10, 6))
        plt.hist(q_values, bins=50)
        plt.xlabel('Q-value')
        plt.ylabel('Frequency')
        plt.title('Q-value Distribution')
        plt.grid(True)
        plt.show()
        
        print(f"Mean Q-value: {np.mean(q_values):.3f}")
        print(f"Std Q-value: {np.std(q_values):.3f}")
        print(f"Min Q-value: {np.min(q_values):.3f}")
        print(f"Max Q-value: {np.max(q_values):.3f}")
    
    @staticmethod
    def check_gradient_flow(model):
        """检查梯度流"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        print(f"Total gradient norm: {total_norm:.6f}")
        
        if total_norm < 1e-6:
            print("WARNING: Gradients are vanishing!")
        elif total_norm > 100:
            print("WARNING: Gradients are exploding!")
    
    @staticmethod
    def visualize_policy(env, policy, num_episodes=5):
        """可视化策略"""
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                env.render()
                action = policy.select_action(state, deterministic=True)
                state, reward, done, _ = env.step(action)
                total_reward += reward
            
            print(f"Episode {episode + 1} Reward: {total_reward}")
        
        env.close()
    
    @staticmethod
    def log_training_stats(episode, metrics):
        """记录训练统计"""
        print(f"\n=== Episode {episode} ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
```

---

## 十五、高级主题

### 15.1 层次强化学习 (HRL)

```python
class HierarchicalPolicy:
    """
    层次策略：高层策略选择子目标，低层策略执行动作
    """
    def __init__(self, state_dim, action_dim, goal_dim):
        # 高层策略（元控制器）
        self.high_level = PolicyNetwork(state_dim, goal_dim)
        
        # 低层策略（控制器）
        self.low_level = PolicyNetwork(state_dim + goal_dim, action_dim)
        
        self.high_optimizer = optim.Adam(self.high_level.parameters(), lr=1e-4)
        self.low_optimizer = optim.Adam(self.low_level.parameters(), lr=3e-4)
    
    def select_goal(self, state):
        """高层策略选择子目标"""
        state = torch.FloatTensor(state).unsqueeze(0)
        goal_probs = self.high_level(state)
        dist = Categorical(goal_probs)
        goal = dist.sample()
        return goal.item(), dist.log_prob(goal)
    
    def select_action(self, state, goal):
        """低层策略选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        goal_tensor = torch.FloatTensor([goal]).unsqueeze(0)
        combined = torch.cat([state, goal_tensor], dim=-1)
        
        action_probs = self.low_level(combined)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def train_step(self, trajectory):
        """
        训练层次策略
        trajectory: [(state, goal, action, reward, next_state, done)]
        """
        # 训练低层策略
        low_loss = 0
        for state, goal, action, reward, next_state, done in trajectory:
            _, log_prob = self.select_action(state, goal)
            low_loss += -log_prob * reward  # 简化版，实际应使用优势函数
        
        self.low_optimizer.zero_grad()
        low_loss.backward()
        self.low_optimizer.step()
        
        # 训练高层策略（基于子目标完成情况）
        # 实现省略，取决于具体的子目标定义

class OptionCritic:
    """
    Option-Critic：学习选项（option）的框架
    选项 = (初始集合, 策略, 终止条件)
    """
    def __init__(self, state_dim, num_options, action_dim):
        self.num_options = num_options
        
        # 选项内策略 π(a|s,o)
        self.intra_option_policy = nn.ModuleList([
            PolicyNetwork(state_dim, action_dim) 
            for _ in range(num_options)
        ])
        
        # 选项终止函数 β(s,o)
        self.termination = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_options),
            nn.Sigmoid()
        )
        
        # 选项选择策略 π_Ω(o|s)
        self.option_policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_options)
        )
        
        self.optimizer = optim.Adam(
            list(self.intra_option_policy.parameters()) + 
            list(self.termination.parameters()) + 
            list(self.option_policy.parameters()),
            lr=1e-3
        )
```

### 15.2 元强化学习 (Meta-RL)

```python
class MAML:
    """
    Model-Agnostic Meta-Learning for RL
    学习一个好的初始化，使其能快速适应新任务
    """
    def __init__(self, state_dim, action_dim, alpha=0.01, beta=0.001):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.alpha = alpha  # 内循环学习率
        self.beta = beta    # 外循环学习率
        
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=beta)
    
    def inner_loop_update(self, task_data, num_steps=1):
        """
        内循环：在单个任务上快速适应
        """
        # 复制当前参数
        adapted_params = [p.clone() for p in self.policy.parameters()]
        
        for _ in range(num_steps):
            # 在任务数据上计算损失
            loss = self.compute_task_loss(task_data, adapted_params)
            
            # 计算梯度
            grads = torch.autograd.grad(loss, adapted_params, create_graph=True)
            
            # 更新参数
            adapted_params = [p - self.alpha * g for p, g in zip(adapted_params, grads)]
        
        return adapted_params
    
    def meta_update(self, task_batch):
        """
        外循环：元更新
        """
        meta_loss = 0
        
        for task in task_batch:
            # 内循环适应
            train_data, test_data = task
            adapted_params = self.inner_loop_update(train_data)
            
            # 在测试数据上评估
            task_loss = self.compute_task_loss(test_data, adapted_params)
            meta_loss += task_loss
        
        meta_loss /= len(task_batch)
        
        # 元梯度下降
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def compute_task_loss(self, data, params):
        """计算任务损失"""
        # 实现省略，取决于具体任务
        pass

class RL2:
    """
    RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning
    使用RNN作为元学习器
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.rnn = nn.LSTM(
            state_dim + action_dim + 1,  # state + prev_action + reward
            hidden_dim,
            batch_first=True
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self.optimizer = optim.Adam(
            list(self.rnn.parameters()) + 
            list(self.policy_head.parameters()) + 
            list(self.value_head.parameters()),
            lr=1e-3
        )
    
    def forward(self, history, hidden=None):
        """
        前向传播
        history: [batch, seq_len, state_dim + action_dim + 1]
        """
        rnn_out, hidden = self.rnn(history, hidden)
        
        action_probs = F.softmax(self.policy_head(rnn_out), dim=-1)
        values = self.value_head(rnn_out)
        
        return action_probs, values, hidden
    
    def train_on_task_distribution(self, task_sampler, num_iterations=1000):
        """
        在任务分布上训练
        """
        for iteration in range(num_iterations):
            # 采样一批任务
            tasks = task_sampler.sample(batch_size=16)
            
            total_loss = 0
            for task in tasks:
                # 收集任务轨迹
                trajectory = self.collect_trajectory(task)
                
                # 计算损失
                loss = self.compute_loss(trajectory)
                total_loss += loss
            
            # 更新
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
```

### 15.3 离线强化学习

```python
class ConservativeQLearning:
    """
    CQL (Conservative Q-Learning)
    用于离线RL，避免外推误差
    """
    def __init__(self, state_dim, action_dim, alpha=1.0):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=3e-4)
        self.alpha = alpha  # CQL正则化系数
    
    def train_step(self, offline_batch):
        """
        CQL训练步骤
        
        最小化：
        CQL_loss = α * (log_sum_exp Q(s,a) - Q(s,a_data)) + TD_loss
        """
        states, actions, rewards, next_states, dones = offline_batch
        
        # 标准Q-learning目标
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        td_loss = F.mse_loss(current_q, target_q)
        
        # CQL惩罚项
        all_q = self.q_net(states)
        logsumexp_q = torch.logsumexp(all_q, dim=1)
        data_q = all_q.gather(1, actions.unsqueeze(1)).squeeze()
        
        cql_loss = (logsumexp_q - data_q).mean()
        
        # 总损失
        loss = td_loss + self.alpha * cql_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), td_loss.item(), cql_loss.item()

class BehaviorCloning:
    """
    行为克隆：模仿学习的基础方法
    直接从专家演示学习策略
    """
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
    
    def train(self, expert_data, num_epochs=100, batch_size=64):
        """
        训练行为克隆
        expert_data: [(state, action)]
        """
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor([s for s, a in expert_data]),
            torch.LongTensor([a for s, a in expert_data])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for states, actions in dataloader:
                # 预测动作
                action_probs = self.policy(states)
                
                # 交叉熵损失
                loss = F.cross_entropy(action_probs, actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

---

## 十六、实际应用案例

### 16.1 Atari游戏

```python
class AtariDQN:
    """
    在Atari游戏上应用DQN
    """
    def __init__(self, num_actions):
        self.q_net = self.build_cnn(num_actions)
        self.target_net = self.build_cnn(num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(100000)
    
    def build_cnn(self, num_actions):
        """构建CNN网络处理图像"""
        return nn.Sequential(
            # 输入: [batch, 4, 84, 84] (4帧灰度图)
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def preprocess_frame(self, frame):
        """
        预处理帧：
        1. 转灰度
        2. 裁剪
        3. 缩放到84x84
        """
        # 实现省略
        pass
    
    def train(self, env, num_frames=10000000):
        """训练"""
        frame_stack = []  # 保持最近4帧
        
        state = env.reset()
        frame = self.preprocess_frame(state)
        
        for _ in range(4):
            frame_stack.append(frame)
        
        for frame_idx in range(num_frames):
            # 选择动作
            state_tensor = torch.FloatTensor(np.array(frame_stack)).unsqueeze(0)
            action = self.select_action(state_tensor)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 处理帧
            next_frame = self.preprocess_frame(next_state)
            next_frame_stack = frame_stack[1:] + [next_frame]
            
            # 存储经验
            self.replay_buffer.push(
                np.array(frame_stack),
                action,
                reward,
                np.array(next_frame_stack),
                done
            )
            
            # 训练
            if frame_idx > 10000:
                self.train_step()
            
            frame_stack = next_frame_stack
            
            if done:
                state = env.reset()
                frame = self.preprocess_frame(state)
                frame_stack = [frame] * 4
```

### 16.2 机器人控制

```python
class RobotController:
    """
    使用RL控制机器人
    """
    def __init__(self, state_dim, action_dim):
        # 使用SAC（适合连续控制）
        self.agent = SAC(state_dim, action_dim, max_action=1.0)
    
    def train_on_simulation(self, sim_env, num_episodes=1000):
        """在仿真环境中训练"""
        for episode in range(num_episodes):
            state = sim_env.reset()
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = sim_env.step(action)
                
                self.agent.replay_buffer.push(
                    state, action, reward, next_state, done
                )
                
                self.agent.train_step()
                state = next_state
    
    def sim_to_real_transfer(self, real_env, num_episodes=10):
        """
        从仿真到真实的迁移
        使用域随机化和微调
        """
        for episode in range(num_episodes):
            state = real_env.reset()
            done = False
            
            while not done:
                # 使用训练好的策略
                action = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, _ = real_env.step(action)
                
                # 在真实数据上微调
                self.agent.replay_buffer.push(
                    state, action, reward, next_state, done
                )
                
                if len(self.agent.replay_buffer) > 256:
                    self.agent.train_step()
                
                state = next_state
```

### 16.3 推荐系统

```python
class RecommenderAgent:
    """
    基于RL的推荐系统
    """
    def __init__(self, num_items, embedding_dim=64):
        self.num_items = num_items
        
        # 物品嵌入
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # 策略网络
        self.policy = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),  # 用户状态 + 候选物品
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = optim.Adam(
            list(self.item_embeddings.parameters()) + 
            list(self.policy.parameters()),
            lr=1e-3
        )
    
    def get_user_state(self, user_history):
        """从用户历史获取状态表示"""
        # 平均历史物品的嵌入
        history_items = torch.LongTensor(user_history)
        embeddings = self.item_embeddings(history_items)
        return embeddings.mean(dim=0)
    
    def recommend(self, user_state, candidate_items, top_k=10):
        """推荐top-k物品"""
        scores = []
        
        for item in candidate_items:
            item_emb = self.item_embeddings(torch.LongTensor([item]))
            combined = torch.cat([user_state, item_emb.squeeze()], dim=0)
            score = self.policy(combined.unsqueeze(0))
            scores.append(score.item())
        
        # 选择top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [candidate_items[i] for i in top_indices]
    
    def train_step(self, user_history, item, reward):
        """训练步骤"""
        user_state = self.get_user_state(user_history)
        item_emb = self.item_embeddings(torch.LongTensor([item]))
        
        combined = torch.cat([user_state, item_emb.squeeze()], dim=0)
        predicted_score = self.policy(combined.unsqueeze(0))
        
        # 使用实际反馈作为目标
        loss = F.binary_cross_entropy(predicted_score, torch.FloatTensor([[reward]]))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

---

## 十七、总结与未来方向

### 17.1 核心概念总结

```python
class RLSummary:
    """
    强化学习核心概念总结
    """
    
    @staticmethod
    def key_components():
        """关键组成部分"""
        return {
            '智能体': '做出决策的实体',
            '环境': '智能体交互的世界',
            '状态': '环境的描述',
            '动作': '智能体的选择',
            '奖励': '反馈信号',
            '策略': '状态到动作的映射',
            '价值函数': '长期回报的预期'
        }
    
    @staticmethod
    def algorithm_categories():
        """算法分类"""
        return {
            '基于值': ['Q-Learning', 'DQN', 'Double DQN', 'Dueling DQN'],
            '基于策略': ['REINFORCE', 'PPO', 'TRPO'],
            'Actor-Critic': ['A2C', 'A3C', 'SAC', 'TD3', 'DDPG'],
            '基于模型': ['Dyna-Q', 'MBPO', 'World Models'],
            '其他': ['逆强化学习', '元强化学习', '多智能体RL']
        }
    
    @staticmethod
    def when_to_use_what():
        """何时使用哪种算法"""
        return {
            '离散动作空间': 'DQN, PPO, A2C',
            '连续动作空间': 'SAC, TD3, DDPG, PPO',
            '需要样本效率': 'SAC, TD3, 基于模型的方法',
            '大规模并行': 'A3C, IMPALA',
            '离线数据': 'CQL, BCQ, 行为克隆',
            '稀疏奖励': '层次RL, 好奇心驱动, HER',
            '多智能体': 'MADDPG, QMIX, MAPPO'
        }
    
    @staticmethod
    def common_pitfalls():
        """常见陷阱"""
        return [
            '1. 过拟合到训练环境',
            '2. 奖励设计不当导致意外行为',
            '3. 探索不足陷入局部最优',
            '4. 超参数敏感性高',
            '5. 训练不稳定',
            '6. 仿真到真实的差距',
            '7. 非平稳性问题',
            '8. 样本效率低'
        ]
    
    @staticmethod
    def best_practices():
        """最佳实践"""
        return [
            '1. 从简单算法开始（如DQN, PPO）',
            '2. 仔细设计奖励函数',
            '3. 使用经验回放和目标网络',
            '4. 进行充分的探索',
            '5. 监控学习曲线和关键指标',
            '6. 使用标准化和归一化',
            '7. 调整超参数',
            '8. 进行消融实验',
            '9. 在多个随机种子上测试',
            '10. 从专家演示学习（如果可用）'
        ]
```

### 17.2 未来研究方向

```markdown
## 强化学习的未来方向

### 1. 样本效率
- 更高效的探索策略
- 更好的模型学习
- 离线RL的进展
- 迁移学习和元学习

### 2. 泛化能力
- 跨任务泛化
- 零样本学习
- 少样本学习
- 域适应

### 3. 安全性和可靠性
- 安全探索
- 约束强化学习
- 鲁棒性
- 可解释性

### 4. 真实世界应用
- 仿真到真实的迁移
- 人机交互
- 现实世界的约束
- 长期规划

### 5. 理论基础
- 收敛性保证
- 样本复杂度分析
- 探索-利用权衡
- 函数逼近的理论

### 6. 多智能体系统
- 协作与竞争
- 通信学习
- 社会学习
- 涌现行为

### 7. 与其他AI领域结合
- RL + 大语言模型
- RL + 计算机视觉
- RL + 因果推理
- RL + 知识图谱
```

---

## 附录：常用库和资源

```python
class RLResources:
    """强化学习资源"""
    
    @staticmethod
    def popular_libraries():
        """流行的RL库"""
        return {
            'OpenAI Gym': '标准RL环境接口',
            'Stable-Baselines3': 'PyTorch实现的标准算法',
            'RLlib': 'Ray生态的可扩展RL库',
            'TF-Agents': 'TensorFlow的RL库',
            'Tianshou': '模块化的PyTorch RL框架',
            'CleanRL': '简洁的RL实现',
            'Dopamine': 'Google的RL研究框架'
        }
    
    @staticmethod
    def simulation_environments():
        """仿真环境"""
        return {
            'OpenAI Gym': '经典控制、Atari等',
            'MuJoCo': '物理仿真',
            'PyBullet': '开源机器人仿真',
            'Unity ML-Agents': 'Unity游戏引擎',
            'DeepMind Control Suite': '连续控制任务',
            'ProcGen': '程序生成的游戏环境',
            'MinAtar': '简化版Atari',
            'Roboschool': '机器人学习环境'
        }
    
    @staticmethod
    def learning_resources():
        """学习资源"""
        return {
            '书籍': [
                'Reinforcement Learning: An Introduction (Sutton & Barto)',
                'Deep Reinforcement Learning Hands-On (Lapan)',
                'Algorithms for Reinforcement Learning (Szepesvári)'
            ],
            '课程': [
                'David Silver RL Course',
                'CS285 Deep RL (UC Berkeley)',
                'DeepMind x UCL RL Course'
            ],
            '论文': [
                'DQN (Mnih et al., 2015)',
                'PPO (Schulman et al., 2017)',
                'SAC (Haarnoja et al., 2018)',
                'AlphaGo (Silver et al., 2016)'
            ]
        }

# 完整笔记结束
```

---

## 结语

本笔记涵盖了强化学习从基础到高级的核心内容，包括：

✅ **基础理论**：MDP、价值函数、策略
✅ **经典算法**：动态规划、蒙特卡洛、时序差分
✅ **深度强化学习**：DQN、PPO、SAC、TD3等
✅ **高级主题**：多智能体、元学习、离线RL
✅ **实际应用**：游戏、机器人、推荐系统

**建议学习路径**：
1. 掌握基础概念和数学原理
2. 实现简单的表格方法（Q-Learning）
3. 学习深度强化学习算法（DQN, PPO）
4. 在标准环境上实验（Gym）
5. 探索高级主题和实际应用

祝学习顺利！🚀
