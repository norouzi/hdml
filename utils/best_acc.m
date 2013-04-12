function W = best_acc(W_set, which_param)

best_acc = -1;

for j = 1:numel(W_set)
  if (W_set(j).params.acc > best_acc)
    best_acc = W_set(j).params.acc;
    W = W_set(j);
  end
  fprintf('%.1d %.3f\n', eval(['W_set(j).params.', which_param,'(1)']), W_set(j).params.acc);
end
