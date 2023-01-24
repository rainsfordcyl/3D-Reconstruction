function [newvecA, newvecB] = updateAB(vecA, vecB, fd, HAA, HAB, HBB)
% updateAB Updates A and B
%
% [newvecA, newvecB] = updateAB(vecA, vecB, fd, HAA, HAB, HBB) where vecA
% and vecB are vectorized A and B, fd is the first derivative, and HAA,
% HAB, and HBB are the second derivative. Finally, newvecA and newvecB are
% updated A and B.
%
% CSC 262 Final Paper

mr = size(HAA, 1);
nr = size(HAB, 2);
r = size(HBB, 2);

fd1 = fd(1:mr);
fd2 = fd(mr+1:end);

CinvBt = zeros(nr,mr);
Cinvb2 = zeros(nr,1);

for i = 1:nr/r
    offset = (i-1)*r;
    Ci = HBB(offset+1:offset+r,1:r);
    
    Cii = inv(Ci);
    tmp = Cii*HAB(:,offset+1:offset+r)';
    CinvBt(offset+1:offset+r,:) = tmp;
    Cinvb2(offset+1:offset+r) = Cii*fd2(offset+1:offset+r);
end

spCinvBt = sparse(CinvBt);
spA = sparse(HAA);
spB = sparse(HAB);

Pinv = spA - spB*spCinvBt;

x1 = Pinv\(fd1 - HAB*Cinvb2);
x2 = Cinvb2 -  CinvBt*x1; 

x1 = full(x1);
x2 = full(x2);

newvecA = vecA-x1;
newvecB = vecB-x2;
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