function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
y_binary = zeros(m,num_labels);
for i=1:m
  y_binary(i,y(i)) = 1;
end
%disp('size of y_binary');
%disp(size(y_binary));

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%X = [ones(m,1) X];
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m,1) X];
%disp('size of a1, Theta1');
%disp(size(a1));
%disp(size(Theta1));
z2 = Theta1 * a1';
disp('size of z2');
disp(size(z2));
a2 = [ones(1,m); sigmoid(z2)];
%disp('size of a2, Theta2');
%disp(size(a2));
%disp(size(Theta2));
z3 = Theta2 * a2;
%disp('size of z3');
%disp(size(z3));
a3 = sigmoid(z3);
%disp('size of a3');
%disp(size(a3));
h_theta = a3';

j_m = zeros(1,m);
for i=1:m
  j_m(i) = sum( -y_binary(i,:) .* log(h_theta(i,:))...
                - (1 - y_binary(i,:)) .* log(1 - h_theta(i,:)) );
end
J = 1/m * sum(j_m); % non regularized cost function

reg_theta1 = 0;
for j=1:hidden_layer_size
  for k=2:input_layer_size+1 % not regularizing bias term
    reg_theta1 += Theta1(j,k)^2;
  end
end
reg_theta2 = 0;
for j=2:hidden_layer_size+1 % not regularizing bias term
  for k=1:num_labels
    reg_theta2 += Theta2(k,j)^2;
  end
end
reg_term = reg_theta1 + reg_theta2;

J = 1/m * sum(j_m) + lambda/(2 * m) * reg_term; % regularized cost function
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

disp(size(Theta1_grad));
disp(size(y_binary));
disp(size(h_theta));

%y_binary
%h_theta

%m=2;
Delta =
for i=1:m
  delta3 = y_binary(i,:)' - h_theta(i,:)';
  delta2 = Theta2' * delta3 .* [ones(1,m); sigmoidGradient(z2)];
end


Theta1_grad;
Theta2_grad;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
