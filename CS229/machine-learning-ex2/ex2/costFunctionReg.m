function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y);     % number of training examples
n = length(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% cost fucntion
%% cost term
cost_i = zeros(1,m);
for i=1:m
  temp = -y(i)*logm(sigmoid(theta'*X(i,:)')) - (1 - y(i))...
  * logm(1 - sigmoid(theta'*X(i,:)'));
  cost_i(i) = temp;
end
%% regularizing term
reg_n = zeros(1,n);
for j=2:n  %regularization does not account for theta_0
  reg_n(j) = theta(j).^2;
end

J = 1/m*sum(cost_i) + lambda/(2*m)*sum(reg_n);

% gracident function

grad_j = zeros(n,1);
% first term
cost_prime_i = zeros(1,m);
for i=1:m
    temp = (sigmoid(theta'*X(i,:)') - y(i)) * X(i,1);
    cost_prime_i(i) = temp;
  end
grad_j(1) = 1/m*sum(cost_prime_i);

for j=2:n % starting at j >= 1
  cost_prime_i = zeros(1,m);
  for i=1:m
    temp = (sigmoid(theta'*X(i,:)') - y(i)) * X(i,j);
    cost_prime_i(i) = temp;
  end
  J_prime = 1/m*sum(cost_prime_i);
  grad_j(j) = J_prime + lambda/m *theta(j) ;
end
grad = grad_j;



% =============================================================

end
