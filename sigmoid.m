function y = sigmoid(x)
% 计算sigmoid函数的值
y = 1 ./ (1 + exp(-x));
end