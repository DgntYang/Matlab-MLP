function g = sigmoid_gradient(z)
% 计算sigmoid函数的梯度
g = sigmoid(z) .* (1 - sigmoid(z));
end