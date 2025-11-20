注意力机制
自注意力机制（Self-Attention Mechanism）是一种能够捕获序列内部元素之间关系的机制，它通过计算序列中每个元素与其他所有元素的相关性来实现信息的有效整合。其基本思想是将输入序列映射为查询(Query)、键(Key)和值(Value)三个矩阵，然后通过计算查询和键的相似度得到注意力权重，最后将这些权重与值相乘得到输出。 自注意力的计算步骤如下：

计算查询、键和值

其中，是输入序列，、和是可学习的权重矩阵。
计算注意力分数

计算注意力权重

其中，是softmax函数，表达式为。
计算输出

```python
import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """
    计算Query、Key、Value矩阵
    Args:
        X: 输入矩阵，形状为 (seq_len, d_model)
        W_q: Query权重矩阵，形状为 (d_model, d_k)
        W_k: Key权重矩阵，形状为 (d_model, d_k)  
        W_v: Value权重矩阵，形状为 (d_model, d_v)
    Returns:
        Q: Query矩阵，形状为 (seq_len, d_k)
        K: Key矩阵，形状为 (seq_len, d_k)
        V: Value矩阵，形状为 (seq_len, d_v)
    """
    Q = np.dot(X, W_q)  # X @ W_q
    K = np.dot(X, W_k)  # X @ W_k  
    V = np.dot(X, W_v)  # X @ W_v
    return Q, K, V

def self_attention(Q, K, V):
    """
    计算自注意力
    Args:
        Q: Query矩阵，形状为 (seq_len, d_k)
        K: Key矩阵，形状为 (seq_len, d_k)
        V: Value矩阵，形状为 (seq_len, d_v)
    Returns:
        attention_output: 注意力输出，形状为 (seq_len, d_v)
    """
    # 1. 计算注意力分数: Q @ K^T
    scores = np.dot(Q, K.T)  # (seq_len, seq_len)
    
    # 2. 缩放（可选，通常除以sqrt(d_k)）
    d_k = Q.shape[-1]
    scores = scores / np.sqrt(d_k)
    
    # 3. 应用softmax得到注意力权重
    # 对每一行应用softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # 数值稳定性
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 4. 计算加权的Value: attention_weights @ V
    attention_output = np.dot(attention_weights, V)
    
    return attention_output


if __name__ == "__main__":
    X = np.array(eval(input()))
    W_q = np.array(eval(input()))
    W_k = np.array(eval(input()))
    W_v = np.array(eval(input()))
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print(self_attention(Q, K, V))
```

位置编码
```python
import numpy as np


def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return np.array(-1)
    # 初始化位置编码矩阵和对应索引
    pos = np.array(np.arange(position), np.float32)
    ind = np.array(np.arange(d_model), np.float32)
    pos = pos.reshape(position, 1)
    ind = ind.reshape(1, d_model)
    # 计算角度
    def get_angles(pos, i):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angles
    # 计算正弦和余弦
    angle1 = get_angles(pos, ind)
    sine = np.sin(angle1[:, 0::2])
    cosine = np.cos(angle1[:, 1::2])
    # 拼接正弦和余弦
    pos_encoding = np.concatenate([sine, cosine], axis=-1)
    # 转换为16位浮点数
    pos_encoding = np.float16(pos_encoding)
    return pos_encoding


if __name__ == "__main__":
    position, d_model = map(int, input().split())
    print(pos_encoding(position, d_model).tolist())
```