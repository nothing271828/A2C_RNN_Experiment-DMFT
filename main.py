import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import math
# -------------------- 设备选择 --------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f'Using device: {device}')


# -------------------- 可训练 RNN (端到端) --------------------
class TrainableRNN(nn.Module):
    def __init__(self, N, D_in, g=1.5, dt=0.1, activation=torch.tanh, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.N = N
        self.D_in = D_in
        self.g = g
        self.dt = dt
        self.activation = activation

        J_init = torch.randn(N, N) / np.sqrt(N)
        self.J = nn.Parameter(J_init)

        U_init = torch.randn(N, D_in)
        self.U = nn.Parameter(U_init)

        self.x = None

    def reset(self, batch_size=1):
        self.x = torch.zeros(batch_size, self.N, device=self.J.device)

    def step(self, I):
        act = self.activation(self.x)
        recurrent = act @ self.J.T * (self.g / np.sqrt(self.N))
        external = I @ self.U.T
        dx = -self.x + recurrent + external
        self.x = self.x + self.dt * dx
        return self.x, self.activation(self.x)


# -------------------- Actor‑Critic 读出 --------------------
class ActorCriticReadout(nn.Module):
    def __init__(self, N, c_v):
        super().__init__()
        self.N = N
        self.c_v = c_v
        self.actor = nn.Linear(N, 2, bias=False)
        self.critic = nn.Linear(N, 1, bias=False)

    def forward(self, x, phi_x):
        logits = self.actor(x) / self.N
        value = self.critic(phi_x).squeeze(-1) / (self.N * self.c_v)
        return logits, value


class PsiNormalization(nn.Module):
    """
    可配置的策略归一化层。
    保证对读出缩放的（近似）不变性，或提供对照基准。
    方法：
      - 'power': 幂归一化。平移后取 α 次幂再归一化，缩放/平移严格不变。
      - 'layer_norm_softmax': 无仿射 LayerNorm + Softmax（可选温度 τ）。
      - 'softmax': 普通 Softmax，用于对照（缩放敏感）。
    """

    def __init__(self, method='power', **kwargs):
        super().__init__()
        self.method = method

        if method == 'power':
            self.eps = kwargs.get('eps', 1e-8)
            self.alpha = kwargs.get('alpha', 2.0)
        elif method == 'layer_norm_softmax':
            self.eps = kwargs.get('eps', 1e-8)
            self.tau = kwargs.get('tau', 1.0)
        elif method == 'softmax':
            self.tau = kwargs.get('tau', 1.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, logits):
        """
        logits: (batch_size, num_actions)
        返回: probs (batch_size, num_actions)，每行和为 1
        """
        if self.method == 'power':
            # 平移使所有值 ≥ 0，保证幂运算不会出现复数
            m, _ = logits.min(dim=-1, keepdim=True)  # (batch, 1)
            shifted = logits - m + self.eps  # 非负
            # 取幂并归一化
            powered = shifted.pow(self.alpha)  # (batch, num_actions)
            probs = powered / powered.sum(dim=-1, keepdim=True)
            return probs

        elif self.method == 'layer_norm_softmax':
            # 无仿射层归一化：减去均值，除以标准差
            mean = logits.mean(dim=-1, keepdim=True)
            # 使用有偏估计（或不偏均可，只要保持尺度不变）
            var = ((logits - mean) ** 2).mean(dim=-1, keepdim=True)
            std = torch.sqrt(var + self.eps)
            z = (logits - mean) / std
            # 可选的温度缩放
            if self.tau != 1.0:
                z = z / self.tau
            probs = F.softmax(z, dim=-1)
            return probs

        elif self.method == 'softmax':
            # 标准 softmax，可加温度
            if self.tau != 1.0:
                logits = logits / self.tau
            return F.softmax(logits, dim=-1)
        return None


# -------------------- 数据生成 --------------------
def generate_trial(T, D_in, mean_sign, sigma_noise=1.0):
    I_seq = mean_sign + sigma_noise * torch.randn(T, D_in)
    return I_seq


# -------------------- 单条轨迹的损失计算（不更新参数） --------------------
def compute_loss_for_trial(rnn, ac_model, psi, I_seq, mean_sign,
                           gamma=0.9, c_v=1, c_p=1, beta=1):
    """
    给定一条完整的 input 序列和符号，计算一个序列的 actor-critic 损失。
    返回：总损失, 策略损失, 价值损失, 准确率
    """
    batch_size = 1
    T = I_seq.shape[0]
    rnn.reset(batch_size)

    log_probs, values, rewards = [], [], []
    correct = 0
    device = I_seq.device  # 获取当前输入所在设备

    for t in range(T):
        x, phi_x = rnn.step(I_seq[t:t + 1])
        logits, value = ac_model(x, phi_x)
        probs = psi(logits)
        dist = Categorical(probs)
        action = dist.sample()

        target = 1 if mean_sign == 1 else 0
        reward = 1.0 if action.item() == target else 0.0
        if action.item() == target:
            correct += 1

        log_probs.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)

    # 计算回报和优势
    returns, advantages = [], []
    R = torch.zeros(1, device=device)
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R
        returns.insert(0, R)
        next_value = values[t + 1] if t + 1 < T else torch.zeros(1, device=device)
        adv = rewards[t] + gamma * next_value - values[t].detach()
        advantages.insert(0, adv)

    returns = torch.stack(returns).squeeze()
    advantages = torch.stack(advantages).detach().squeeze()
    log_probs = torch.stack(log_probs).squeeze()
    values = torch.stack(values).squeeze()

    policy_loss = -c_p * (log_probs * advantages).mean()
    value_loss = 0.5 * c_v ** 2 * (advantages ** 2).mean()

    l2_loss = (rnn.J.pow(2).sum() + rnn.U.pow(2).sum() +
               ac_model.actor.weight.pow(2).sum() +
               ac_model.critic.weight.pow(2).sum())

    total_loss = N * (policy_loss + value_loss) + l2_loss / (2 * beta)
    accuracy = correct / T
    return total_loss, N * policy_loss, N * value_loss, accuracy


# -------------------- 评估函数 --------------------
def evaluate(rnn, ac_model, psi, T, D_in, sigma_noise, num_trials=200):
    test_acc = 0.0
    device = next(rnn.parameters()).device  # 获取模型所在设备
    with torch.no_grad():
        for _ in range(num_trials):
            mean_sign = 1 if np.random.rand() > 0.5 else -1
            rnn.reset(1)
            I_seq = generate_trial(T, D_in, mean_sign, sigma_noise).to(device)
            correct = 0
            for t in range(T):
                x, phi_x = rnn.step(I_seq[t:t + 1])
                logits, _ = ac_model(x, phi_x)
                probs = psi(logits)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)
                target = 1 if mean_sign == 1 else 0
                if action.item() == target:
                    correct += 1
            test_acc += correct / T
    return test_acc / num_trials


# ==================== 训练函数 (Adam) ====================
def train_A2C(rnn, ac, psi, dataset,
              T_dur=30, D_in=10, sigma_noise=2.0,
              gamma=0.95, c_p=100, c_v=1, beta=100,
              lr=0.01, max_epochs=500, eval_interval=10,
              patience=50, device='cpu'):
    """
    使用 Adam 优化器的 A2C 训练。
    返回: (rnn, ac, train_acc_history, test_acc_history)
    """
    optimizer = torch.optim.Adam(list(rnn.parameters()) + list(ac.parameters()), lr=lr)

    train_acc_history = []
    test_acc_history = []
    best_test_acc = -1.0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_p_loss = 0.0
        epoch_v_loss = 0.0

        for sample in dataset:
            loss, p_loss, v_loss, acc = compute_loss_for_trial(
                rnn, ac, psi,
                sample['I_seq'], sample['mean_sign'],
                gamma=gamma, c_p=c_p, c_v=c_v, beta=beta
            )
            epoch_loss += loss.item()
            epoch_p_loss += p_loss.item()
            epoch_v_loss += v_loss.item()
            epoch_acc += acc
            loss.backward()

        optimizer.step()

        epoch_loss /= len(dataset)
        epoch_acc /= len(dataset)

        if epoch % eval_interval == 0 or epoch == 1:
            test_acc = evaluate(rnn, ac, psi, T_dur, D_in, sigma_noise, num_trials=200)
            train_acc_history.append(epoch_acc)
            test_acc_history.append(test_acc)
            print(f"Epoch {epoch:4d}: Train Loss={epoch_loss:.4f}, "
                  f"Policy Loss={epoch_p_loss/len(dataset):.4f}, "
                  f"Value Loss={epoch_v_loss/len(dataset):.4f}, "
                  f"Train Acc={epoch_acc:.3f}, Test Acc={test_acc:.4f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += eval_interval

            if epochs_no_improve >= patience or test_acc > 0.99:
                break

    return rnn, ac, train_acc_history, test_acc_history


# ==================== 训练函数 (Langevin Dynamics) ====================
def train_Langevin(rnn, ac, psi, dataset,
                   T_dur=30, D_in=10, sigma_noise=2.0,
                   gamma=0.95, c_p=100, c_v=1, beta=100,
                   lr=0.01, max_epochs=500, eval_interval=10,
                   patience=50, device='cpu'):
    """
    使用郎之万动力学 (SGLD) 的训练算法。
    更新公式: ΔΘ = - lr * ∇E(Θ) + sqrt(2 * lr / beta) * ξ
    返回: (rnn, ac, train_acc_history, test_acc_history)
    """
    params = list(rnn.parameters()) + list(ac.parameters())

    train_acc_history = []
    test_acc_history = []
    best_test_acc = -1.0
    epochs_no_improve = 0
    dataset_size = len(dataset)

    for epoch in range(1, max_epochs + 1):
        # 清零所有梯度
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_p_loss = 0.0
        epoch_v_loss = 0.0

        # 累加每个样本的损失，最后取平均再反向传播
        total_loss_accum = 0.0
        for sample in dataset:
            loss, p_loss, v_loss, acc = compute_loss_for_trial(
                rnn, ac, psi,
                sample['I_seq'], sample['mean_sign'],
                gamma=gamma, c_p=c_p, c_v=c_v, beta=beta
            )
            epoch_loss += loss.item()
            epoch_p_loss += p_loss.item()
            epoch_v_loss += v_loss.item()
            epoch_acc += acc
            total_loss_accum += loss  # 累加张量

        # 平均损失，反向传播得到平均梯度
        mean_loss = total_loss_accum / dataset_size
        mean_loss.backward()

        # 朗之万更新: Θ <- Θ - lr * ∇E(Θ) + sqrt(2 * lr / beta) * ξ
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    noise = torch.randn_like(p) * math.sqrt(2 * lr / beta)
                    p.add_(-lr * p.grad + noise)

        epoch_loss /= dataset_size
        epoch_acc /= dataset_size

        if epoch % eval_interval == 0 or epoch == 1:
            test_acc = evaluate(rnn, ac, psi, T_dur, D_in, sigma_noise, num_trials=200)
            train_acc_history.append(epoch_acc)
            test_acc_history.append(test_acc)
            print(f"Epoch {epoch:4d}: Train Loss={epoch_loss:.4f}, "
                  f"Policy Loss={epoch_p_loss/dataset_size:.4f}, "
                  f"Value Loss={epoch_v_loss/dataset_size:.4f}, "
                  f"Train Acc={epoch_acc:.3f}, Test Acc={test_acc:.4f}")

            # if test_acc > best_test_acc:
            #     best_test_acc = test_acc
            #     epochs_no_improve = 0
            # else:
            #     epochs_no_improve += eval_interval
            #
            # if epochs_no_improve >= patience or test_acc > 0.99:
            #     break

    return rnn, ac, train_acc_history, test_acc_history


# ==================== 绘图函数 ====================
def plot_training_curves(train_acc_history, test_acc_history, eval_interval):
    """
    绘制训练与测试准确率曲线。
    """
    eval_epochs = list(range(1, len(train_acc_history) * eval_interval + 1, eval_interval))
    plt.figure(figsize=(10, 4))
    plt.plot(eval_epochs, train_acc_history, marker='o', label='Train Acc')
    plt.plot(eval_epochs, test_acc_history, marker='s', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Training')
    plt.legend()
    plt.grid(True)
    plt.show()


# ==================== 主程序示例 ====================
if __name__ == "__main__":
    # 超参数
    N = 100
    g = 1.5
    dt = 0.1
    c_p = 1
    c_v = 1
    gamma_ = 0.95
    D_in = 10
    T_dur = 30
    sigma_noise = 2.0
    lr = 0.001
    max_epochs = 100
    eval_interval = 1
    patience = 50
    beta = 10000

    seed_everything = 42
    torch.manual_seed(seed_everything)
    np.random.seed(seed_everything)

    # 生成固定数据集
    data_seed = 12345
    torch.manual_seed(data_seed)
    dataset = [
        {'I_seq': generate_trial(T_dur, D_in, 1, sigma_noise), 'mean_sign': 1},
        {'I_seq': generate_trial(T_dur, D_in, -1, sigma_noise), 'mean_sign': -1}
    ]
    for sample in dataset:
        sample['I_seq'] = sample['I_seq'].to(device)

    torch.manual_seed(seed_everything)

    # 初始化模型
    rnn = TrainableRNN(N, D_in, g=g, dt=dt, seed=seed_everything).to(device)
    ac = ActorCriticReadout(N, c_v).to(device)
    psi = PsiNormalization(method='layer_norm_softmax', tau=1)

    # # --- 使用 Adam 训练 ---
    # print("\n===== Training with Adam =====")
    # rnn_adam, ac_adam, train_hist_adam, test_hist_adam = train_A2C(
    #     rnn, ac, psi, dataset,
    #     T_dur=T_dur, D_in=D_in, sigma_noise=sigma_noise,
    #     gamma=gamma_, c_p=c_p, c_v=c_v, beta=beta,
    #     lr=lr, max_epochs=max_epochs, eval_interval=eval_interval,
    #     patience=patience, device=device
    # )
    # plot_training_curves(train_hist_adam, test_hist_adam, eval_interval)

    # --- 重新初始化，使用 Langevin 训练 ---
    torch.manual_seed(seed_everything)
    rnn2 = TrainableRNN(N, D_in, g=g, dt=dt, seed=seed_everything).to(device)
    ac2 = ActorCriticReadout(N, c_v).to(device)

    print("\n===== Training with Langevin Dynamics =====")
    rnn_lang, ac_lang, train_hist_lang, test_hist_lang = train_Langevin(
        rnn2, ac2, psi, dataset,
        T_dur=T_dur, D_in=D_in, sigma_noise=sigma_noise,
        gamma=gamma_, c_p=c_p, c_v=c_v, beta=beta,
        lr=lr, max_epochs=max_epochs, eval_interval=eval_interval,
        patience=patience, device=device
    )
    plot_training_curves(train_hist_lang, test_hist_lang, eval_interval)