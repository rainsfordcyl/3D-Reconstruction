function [HAA, HAB, HBB] = secondDerivative(O, W, A, B)
% secondDerivative Finds the second derivative, which is the Hessian, of
% the function F
%
% [HAA, HAB, HBB] = secondDerivative(O, W, A, B) where O is a matrix to be
% factorized, W is a weight matrix, and A and B are initial facotrized
% matrices. Finally, HAA, HAB, and HBB are the parts of the Hessian matrix.
%
% CSC 262 Final Paper

% Find the size of A and B
m = size(O,1);
n = size(O,2);
r = size(A,2);

% Initialize the Hessian matrix
HAA = zeros(m*r,m*r);
HAB = zeros(m*r,n*r);
HBB = zeros(n*r,r);

% Calculate HAA
for a = 1:m
    for b = 1:r
        second_index = (a-1)*r+b;
        for f = 1:r
            first_index = (a-1)*r+f;
            for j = 1:n
                HAA(first_index,second_index) = HAA(first_index,second_index) + W(a,j)*W(a,j)*B(j,b)*B(j,f);
            end
            HAA(first_index,second_index) = HAA(first_index,second_index) * 2;
        end
    end
end

% Calculate HAB
for c = 1:n
    for d = 1:r
    second_index = (c-1)*r+d;
        for e = 1:m
            for f = 1:r
                first_index = (e-1)*r+f;
                AepBcp_sum = 0;
                if d==f
                    for p = 1:r
                        AepBcp_sum = AepBcp_sum + A(e,p)*B(c,p);  
                    end
                    AepBcp_sum = AepBcp_sum - O(e,c);  
                end
                HAB(first_index,second_index) = 2*W(e,c)*W(e,c)*(A(e,d)*B(c,f) + AepBcp_sum); 
            end
        end
    end
end

% Calculate HBB
for c = 1:n
    for d = 1:r
        second_index = d;
        for h = 1:r
            first_index = (c-1)*r+h;
            for i = 1:m
                HBB(first_index,second_index) = HBB(first_index,second_index) + W(i,c)*W(i,c)*A(i,d)*A(i,h);
            end
            HBB(first_index,second_index) = HBB(first_index,second_index) * 2;
        end
    end
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