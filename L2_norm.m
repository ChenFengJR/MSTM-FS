function X_norm = L2_norm(X, mode)
% Normalize each 'mode' of X into unit 
% mode: 'row'(default) or 'col';
% X_norm = X;
% return;
% error('!');
eps = 0.0001;
if nargin < 2, mode = 'row'; end
switch mode
    case 'row', X_norm = X./repmat(sqrt(sum(X.^2, 2) + eps), 1, size(X, 2));
    case 'col', X_norm = X./repmat(sqrt(sum(X.^2, 1) + eps), size(X, 1), 1);
    otherwise,  error('Wrong L2 norm mode.');
end

end