function fd = firstDerivative(O, W, A, B)
% firstDerivative Finds the first derivative, which is the gradient, of the
% function F
%
% fd = firstDerivative(O, W, A, B) where O is a matrix to be factorized, W
% is a weight matrix, and A and B are initial facotrized matrices. Finally,
% fd is the first deriviate, which is the gradient, of the function F.
%
% CSC 262 Final Paper

% Find the size of A and B
m = size(O,1);
n = size(O,2);
r = size(A, 2);

% Initalize the first derivative
fd = zeros((m+n)*r,1);

% Calculate the first derivative based on the paper "Damped Newton
% Algorithms for Matrix Factorization with Missing Data"
for a = 1:m
    bww = B.*W(a,:)'.*W(a,:)';
    bam = sum(B.*A(a,:), 2) - O(a,:)';
    fd((a-1)*r+1:a*r) = 2*sum(bww'.*bam', 2);
end

for b = 1:n
    aww = A.*W(:,b)'.*W(:,b)';
    abm = sum(A.*B(b,:), 2) - O(:,b);
    fd(m*r+(b-1)*r+1:m*r+b*r) = 2*sum(aww.*abm, 1);
end

end

%% Acknowledgements
% The idea for matrix factorization with the damped Newton method is from
% the paper "Damped Newton Algorithms for Matrix Factorization with Missing
% Data"
%
% The matrix factorization code is obtained from the author's code on his
% website: https://www.robots.ox.ac.uk/~amb/, and we modified the code on
% our purpose based on our understanding of the paper.
%
% Due to the huge size of the Hessian matrix, the Hessian matrix is solved
% in a sparse way. Our understanding of the sparse Hessian matrix is from
% the author's code and the link:
% https://opus.uleth.ca/bitstream/handle/10133/4601/SULTANA_MARZIA_MSC_2016.pdf