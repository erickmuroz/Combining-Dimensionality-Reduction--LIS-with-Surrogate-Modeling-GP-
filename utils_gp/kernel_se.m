function K = kernel_se(XA, XB, ell, sf2)
%% KERNEL_SE  Squared-exponential (RBF) kernel, isotropic, multi-dim input.
%  k(a,b) = sf2 * exp( -||a-b||^2 / (2 ell^2) ),  R&W (2006) eq. (4.9).
%
%  Inputs:
%    XA   (nA x d) inputs (rows = points, cols = dimensions)
%    XB   (nB x d) inputs
%    ell  scalar length scale (shared across dimensions)
%    sf2  signal variance
%  Output:
%    K    (nA x nB) kernel matrix
%
%  Squared distance via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, clamped at 0
%  for numerical safety. Same algebraic trick generalizes the 1D form to
%  Euclidean distance in d dimensions (used for r = 1, 2, 3).

sa = sum(XA.^2, 2);            % (nA x 1)
sb = sum(XB.^2, 2);            % (nB x 1)
D2 = sa + sb' - 2*(XA*XB');    % (nA x nB)
D2 = max(D2, 0);
K  = sf2 * exp(-0.5 * D2 / ell^2);
end
