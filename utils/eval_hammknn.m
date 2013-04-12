% Evaluates hamming kNN classifier using 1 upto K nearest neighbors for prediction

function [acc acc2] = eval_hammknn(data, W, K, nonlinearity)

Ntraining = data.Ntraining;
Xtraining = data.Xtraining;
Ltraining = double(data.Ltraining);

Ntest = data.Ntest;
Xtest = data.Xtest;
Ltest = double(data.Ltest);

if (~iscell(W))
  B1 = W * Xtraining;
  B2 = W * Xtest;
else
  resp1 = compute_NN_output(W, Xtraining, nonlinearity);
  B1 = resp1{end};
  resp2 = compute_NN_output(W, Xtest, nonlinearity);
  B2 = resp2{end};
end

B1 = logical(single(B1 > 0));
B1 = compactbit(B1);

B2 = logical(single(B2 > 0));
B2 = compactbit(B2);

[nw1 n1] = size(B1);
[nw2 n2] = size(B2);
nb = nw1*8;

if (nw1 ~= nw2)
  error('nw1 ~= nw2\n');
end

if ~exist('K', 'var')
  K = 3;
end

% Assuming there are only 10 labels
% TODO: change if you have more labels
nlabels = 10;

[ret ret2] = hammknn_mex(nlabels, B1, uint32(Ltraining), B2, uint32(Ltest), n1, nb, K);

acc = double(ret) / Ntest;
acc2 = double(ret2) / Ntraining;
