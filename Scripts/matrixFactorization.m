function [A, B] = matrixFactorization(O, W, A, B, iteration, lambda, lambdaStep, threshold)
% matrixFactorization Factorizes an input matrix with the damped Newton
% method
%
% newO = matrixFactorization(O, W, A, B, iteration, lambda, lambdaStep,
% threshold) where O is a matrix to be factorized, W is a weight matrix, A
% and B are initial facotrized matrices, iteration is the maximum number of
% iterations, lambda is the adaptive regularizing parameter, and lambdaStep
% is the paprameter indicating the adaptiveness of lambda, and threshold is
% the threshold for convergence. Finally, A and B are the updated
% factorized matrix based on the damped Newton method.
%
% CSC 262 Final Paper

% Find the size of A and B
m = size(O,1);
n = size(O,2);
r = min(m,n);

% Initialize the previous and current errors
prevError = inf;
currError = norm(W.*(O-A*B'),'fro')^2;

% Initialize count
count = 2;

% Loop until the convergence happens
while count <= iteration && prevError - currError > threshold

    % Vectorize A and B
    vecA = reshape(A',m*r,1);
    vecB = reshape(B',n*r,1);

    % Calculate the start error
    prevError = norm(W.*(O-A*B'),'fro')^2;
    
    % Set an initial lambda
    lambda = lambda/(lambdaStep^2);

    % Calcuate the first and second derivatives
    fd = firstDerivative(O, W, A, B);
    [HAA, HAB, HBB] = secondDerivative(O, W, A, B);

    % Initalize the current error
    currError = -inf;
    while prevError - currError < -threshold
        % Increase lambda
        lambda = lambda*lambdaStep;

        % Calculate the Hessian matrix plus I*lambda
        HAAI = HAA + lambda*eye(m*r);
        HBBI = HBB;
	    for i = 1:r
		    index = [i:r:n*r] + (i-1)*n*r;
		    HBBI(index) = HBBI(index) + lambda;
        end
    
        % Update A and B
        [newvecA, newvecB] = updateAB(vecA, vecB, fd, HAAI, HAB, HBBI);
    
        % Unvectorize A and B
        A = reshape(newvecA,r,m)';
        B = reshape(newvecB,r,n)';
    
        % Update the current error
        currError = norm(W.*(O-A*B'),'fro')^2;
    end
    
    % Increment count
    count = count + 1;
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