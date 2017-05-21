function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta

cost_i = zeros(1,m);
for i=1:m
  temp = -y(i)*logm(sigmoid(theta'*X(i,:)')) - (1 - y(i))...
  * logm(1 - sigmoid(theta'*X(i,:)'));
  cost_i(i) = temp;
end
J = 1/m*sum(cost_i);

grad_j = zeros(n,1);
for j=1:n
  cost_prime_i = zeros(1,m);
  for i=1:m
    temp = (sigmoid(theta'*X(i,:)') - y(i)) * X(i,j);
    cost_prime_i(i) = temp;
  end
  J_prime = 1/m*sum(cost_prime_i);
  grad_j(j) = J_prime;
end
grad = grad_j;
%J = 1/m*sum(-y*log(sigmoid(theta'*X') ) - (1 - y ) ...
%  * log(1 - sigmoid(theta'*X') ));
%
%function grad_i = f_grad(theta)
%end

%1/m sum( (sigmoid(theta'*X') - y)*X')

%for j = 1:size(theta)
%temp_sum = zeros(1,m);
%for i=1:m
%  temp_var = ( (sigmoid(theta*X(i,:))-y(i,:))*X(i,j)' )(1,:);
%  temp_sum(i) = sum(temp_var);%(sigmoid(theta*X(i,:))-y(i,:))*X(i,j)'
%end
%grad(j) = 1/m*sum(temp_sum)
%end


%grad =  1/m*((sigmoid(theta'*X') - y)'*X)





% =============================================================

end

initial_theta = zeros(n + 1, 1);
[cost, grad] = costFunction(initial_theta, X, y);

test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);






























