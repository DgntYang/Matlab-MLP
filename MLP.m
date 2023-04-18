clc;clear;close;

% 加载数据集，具体情况具体分析
data = xlsread("data.xlsx", "A1:B200");
x_train = data(1:150, 1);
y_train = data(1:150, 2);
x_test = data(151:200, 1);
y_test = data(151:200, 2);


% 设置MLP的参数
input_size = 1; % 输入层大小
hidden_size = 50; % 隐藏层大小
output_size = 1; % 输出层大小
lr = 0.001; % 学习率
epochs = 10000; % 训练轮次
interval = 1;
randn("seed", 42)

% 初始化权重和偏置
w1 = randn(input_size, hidden_size); % 输入层到第一个隐藏层的权重矩阵
b1 = zeros(1, hidden_size); % 第一个隐藏层的偏置向量
w2 = randn(hidden_size, hidden_size); % 第一个隐藏层到第二个隐藏层的权重矩阵
b2 = zeros(1, hidden_size); % 第二个隐藏层的偏置向量
w3 = randn(hidden_size, output_size); % 第二个隐藏层到输出层的权重矩阵
b3 = zeros(1, output_size); % 输出层的偏置向量

% 存储损失值
train_losses = [];
test_losses = [];
earlystop_count = 0; % 早期停止计数器
patience = 100;
thereshold = 0.0000001; % 早期停止阈值


% 训练模型
for epoch = 1:epochs % 迭代1000次
    % 前向传播
    z1 = x_train*w1 + b1; % 第一层输入
    a1 = sigmoid(z1); % 第一层激活
    z2 = a1*w2 + b2; % 第二层输入
    a2 = sigmoid(z2); % 第二层激活
    z3 = a2*w3 + b3; % 输出层输入
    y_pred = z3; % 输出层激活

    % 计算损失和梯度
    train_loss = 0.5*mean((y_pred-y_train).^2); % 均方误差损失函数
    dL_dy_pred = y_pred - y_train; % 损失对输出层的导数
    dL_dz3 = dL_dy_pred; % 输出层输入对损失的导数
    dL_da2 = dL_dz3*w3'; % 第二个隐藏层输出对损失的导数
    dL_dz2 = dL_da2.*sigmoid_gradient(z2); % 第二个隐藏层输入对损失的导数
    dL_da1 = dL_dz2*w2'; % 第一个隐藏层输出对损失的导数
    dL_dz1 = dL_da1.*sigmoid_gradient(z1); % 第一个隐藏层输入对损失的导数

    % 反向传播更新权重和偏置
    w3 = w3 - lr*a2'*dL_dz3; % 更新第二个隐藏层到输出层的权重
    b3 = b3 - lr*mean(dL_dz3, 1); % 更新输出层的偏置

    dL_dz2_mean = mean(dL_dz2, 1); % 计算第二个隐藏层的导数的平均值
    w2 = w2 - lr*a1'*dL_dz2; % 更新第一个隐藏层到第二个隐藏层的权重
    b2 = b2 - lr*dL_dz2_mean; % 更新第二个隐藏层的偏置

    w1 = w1 - lr*x_train'*dL_dz1; % 更新输入层到第一个隐藏层的权重
    b1 = b1 - lr*mean(dL_dz1, 1); % 更新第一个隐藏层的偏置
    
    
    % 开始测试并记录损失
    z1 = x_test*w1 + b1; % 第一层输入
    a1 = sigmoid(z1); % 第一层激活
    z2 = a1*w2 + b2; % 第二层输入
    a2 = sigmoid(z2); % 第二层激活
    z3 = a2*w3 + b3; % 输出层输入
    y_pred = z3; % 输出层激活    
    test_loss = 0.5*mean((y_pred-y_test).^2);
    if mod(epoch, interval) == 0
        train_losses = [train_losses, train_loss];
        test_losses = [test_losses, test_loss];
    end
    
    
    % 打印损失
    if mod(epoch, interval) == 0 % 每100次迭代打印一次
        disp(['Epoch   ', num2str(epoch), ', train loss: ', num2str(train_loss), ',   test loss: ', num2str(test_loss)]);
    end
    
    % 早期停止
    n = length(test_losses);
    if epoch > interval
        if abs(test_losses(n) - test_losses(n-1)) < thereshold
            earlystop_count = earlystop_count + 1;
            if earlystop_count > patience % 如果小步长迈进次数超过了耐心值，则早期停止
                disp(['Earlystop at epoch ', num2str(epoch), ' !']);
                epochs = epoch;
                break
            end
        end
    end
    
end


% 对训练集预测
[a, b] = size(x_train);
xmin = min(x_train);
xmax = max(x_train);
ypreds = []; % 存储预测值
for x = linspace(xmin, xmax, a)
    z1 = x*w1 + b1; % 第一层输入
    a1 = sigmoid(z1); % 第一层激活
    z2 = a1*w2 + b2; % 第二层输入
    a2 = sigmoid(z2); % 第二层激活
    z3 = a2*w3 + b3; % 输出层输入
    y_pred = z3; % 输出层激活   
    ypreds = [ypreds, y_pred];
end

% 画训练集loss图
figure(1);
plot(1:interval:epochs, train_losses, 'b', 'linewidth',1);
hold on;
plot(1:interval:epochs, test_losses, 'r', 'linewidth',1);
title("loss")
legend("train loss", "test loss")
xlabel("Epoch")
ylabel("Mse")


% 画训练集拟合图
figure(2);
plot(x_train, y_train, 'b.');
hold on
for i = 1:a
    plot(linspace(xmin, xmax, a), ypreds, 'r', 'linewidth',1)
end
legend("real data", "fit curve")
title("Train")
xlabel("x")

% 画测试集拟合图（数据有限，用验证集代替）
[a, b] = size(x_test);
xmin = min(x_test);
xmax = max(x_test);
ypreds = [];
for x = linspace(xmin, xmax, a)
    z1 = x*w1 + b1; % 第一层输入
    a1 = sigmoid(z1); % 第一层激活
    z2 = a1*w2 + b2; % 第二层输入
    a2 = sigmoid(z2); % 第二层激活
    z3 = a2*w3 + b3; % 输出层输入
    y_pred = z3; % 输出层激活
    ypreds = [ypreds, y_pred];
end

% 画测试集拟合图
figure(3);
plot(x_test, y_test, 'b.');
hold on
for i = 1:a
    plot(linspace(xmin, xmax, a), ypreds, 'r', 'linewidth',1)
end
legend("real data", "fit curve")
title("Test")
xlabel("x")



