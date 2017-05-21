function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
len_i = size(z)(1);
len_j = size(z)(2);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for i=1:len_i
  for j=1:len_j
    g(i,j) = 1/(1+exp(-z(i,j)) );
  end
end
%g = ones(size(z))./(1+exp(z) );
% =============================================================

end


