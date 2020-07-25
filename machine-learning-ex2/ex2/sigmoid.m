function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
l = size(z);
row = l(1);
col = l(2);
for i=1:row
  for j=1:col
    g(i,j) = 1/(1 + exp(-1 * z(i,j)));
  endfor
endfor





% =============================================================

end
