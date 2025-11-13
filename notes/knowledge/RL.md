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
        """单步训练"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样批量经验
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 当前Q值
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 目标Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算损失
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
    
    def train(self, env, num_episodes=1000):
        """训练DQN"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # 选择动作
                action = self.select_action(state)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                # 存储经验
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # 训练
                loss = self.train_step()
                
                state = next_state
            
            # ε衰减
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            episode_rewards.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards
```

### 9.2 Double DQN

```python
class DoubleDQN(DQN):
    """
    Double DQN：解决Q值过估计问题
    
    使用在线网络选择动作，目标网络评估Q值
    Q_target = r + γ * Q_target(s', argmax_a Q(s',a))
    """
    def train_step(self):
        """Double DQN训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 当前Q值
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN目标
        with torch.no_grad():
            # 使用在线网络选择动作
            next_actions = self.q_net(next_states).argmax(1)
            # 使用目标网络评估Q值
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
```

### 9.3 Dueling DQN

```python
class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN网络
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQNNetwork, self).__init__()
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势流 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 结合：Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values
```

### 9.4 Prioritized Experience Replay

```python
class SumTree:
    """求和树：用于优先级采样"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def update(self, idx, priority):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def add(self, priority, data):
        """添加数据"""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def get(self, s):
        """根据优先级采样"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
    
    def _retrieve(self, idx, s):
        """检索叶节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    @property
    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    """
    优先级经验回放
    根据TD误差分配采样优先级
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样权重
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.epsilon = 0.01
    
    def push(self, state, action, reward, next_state, done):
        """添加经验（使用最大优先级）"""
        data = Transition(state, action, reward, next_state, done)
        priority = self.max_priority
        self.tree.add(priority, data)
    
    def sample(self, batch_size):
        """根据优先级采样"""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / batch_size
        
        # 增加β
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        
        # 计算重要性采样权重
        sampling_probs = np.array(priorities) / self.tree.total_priority
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weights /= is_weights.max()
        
        # 转换为张量
        transitions = Transition(*zip(*batch))
        states = torch.FloatTensor(transitions.state)
        actions = torch.LongTensor(transitions.action)
        rewards = torch.FloatTensor(transitions.reward)
        next_states = torch.FloatTensor(transitions.next_state)
        dones = torch.FloatTensor(transitions.done)
        is_weights = torch.FloatTensor(is_weights)
        
        return states, actions, rewards, next_states, dones, idxs, is_weights
    
    def update_priorities(self, idxs, td_errors):
        """更新优先级"""
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries
```

---

## 十、高级算法

### 10.1 PPO (Proximal Policy Optimization)

```python
class PPO:
    """
    PPO：近端策略优化
    
    使用裁剪目标函数限制策略更新幅度
    L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
    
    其中 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 gae_lambda=0.95, epochs=10, batch_size=64):
        
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size
    
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = self.actor_critic(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones):
        """
        计算GAE (Generalized Advantage Estimation)
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """PPO更新"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多次更新
        for _ in range(self.epochs):
            # 随机打乱
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 前向传播
                action_probs, values = self.actor_critic(batch_states)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 总损失
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
    
    def train(self, env, num_episodes=1000, max_steps=1000):
        """训练PPO"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            
            state = env.reset()
            
            for step in range(max_steps):
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                
                state = next_state
                
                if done:
                    break
            
            # 计算回报和优势
            advantages = self.compute_gae(rewards, values, dones)
            returns = [adv + val for adv, val in zip(advantages, values)]
            
            # 更新策略
            self.update(states, actions, log_probs, returns, advantages)
            
            episode_rewards.append(sum(rewards))
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
        
        return episode_rewards
```

### 10.2 DDPG (Deep Deterministic Policy Gradient)

```python
class DDPGActor(nn.Module):
    """DDPG Actor网络（确定性策略）"""
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(DDPGActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action
    
    def forward(self, state):
        return self.max_action * self.network(state)

class DDPGCritic(nn.Module):
    """DDPG Critic网络（Q函数）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DDPGCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))

class DDPG:
    """
    DDPG：深度确定性策略梯度
    适用于连续动作空间
    
    关键技术：
    1. Actor-Critic架构
    2. 目标网络
    3. 经验回放
    4. OU噪声探索
    """
    def __init__(self, state_dim, action_dim, max_action,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 buffer_size=100000, batch_size=64):
        
        # Actor网络
        self.actor = DDPGActor(state_dim, action_dim, max_action)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic网络
        self.critic = DDPGCritic(state_dim, action_dim)
        self.critic_target = DDPGCritic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
    
    def select_action(self, state, noise=0.1):
        """选择动作（带噪声）"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        
        # 添加探索噪声
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def train_step(self):
        """训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        
        # 采样
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 更新Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()
    
    def soft_update(self, source, target):
        """软更新：θ' ← τθ + (1-τ)θ'"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

### 10.3 SAC (Soft Actor-Critic)

```python
class SAC:
    """
    SAC：软Actor-Critic
    最大化熵正则化的目标：J = E[Σ(r_t + α*H(π(·|s_t)))]
    
    关键特点：
    1. 最大熵强化学习
    2. 自动温度调节
    3. 双Q网络（减少过估计）
    """
    def __init__(self, state_dim, action_dim, max_action,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, auto_entropy=True):
        
        # Actor网络（输出高斯分布参数）
        self.actor = GaussianPolicy(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 双Critic网络
        self.critic1 = DDPGCritic(state_dim, action_dim)
        self.critic2 = DDPGCritic(state_dim, action_dim)
        self.critic1_target = DDPGCritic(state_dim, action_dim)
        self.critic2_target = DDPGCritic(state_dim, action_dim)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # 温度参数
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        if auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(100000)
        self.batch_size = 256
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        if deterministic:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        
        return action.detach().numpy()[0]
    
    def train_step(self):