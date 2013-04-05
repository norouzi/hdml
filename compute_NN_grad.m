function grad = compute_NN_grad(W, X, resp, hout, nonlinearity, nonzero_grad)

% nonlinearity = 1 -- hyperbolic tangent
% nonlinearity = 2 -- logistic sigmoid (-0.5 last layer)
% nonlinearity = 3 -- logistic sigmoid + hyperbolic tangent (last layer)

ncases = size(resp{1},2);
if isa(X, 'gsingle')
  gputype = 'gsingle';
  onesncases = gones(ncases,1,gputype);
elseif isa(X, 'gdouble')
  gputype = 'gsingle';
  onesncases = gones(ncases,1,gputype);
else
  onesncases = ones(ncases,1);
end

nlayer = numel(W);

if (nonlinearity == 1)
  
  if (exist('nonzero_grad', 'var'))
    snonzero_grad = sum(nonzero_grad);
    if (snonzero_grad == 0)
      for i=nlayer:-1:1
	grad{i} = zeros(size(W{i}));
      end
      return;
    end
    
    for i=nlayer:-1:1
      if (i == nlayer)
	tmp = hout(:, nonzero_grad);
      else
	tmp = (W{i+1}' * backward{i+1});
      end
      
      if (size(tmp, 1) ~= size(resp{i}, 1))
	tmp = tmp(1:end-1, :);
      end
    
      backward{i} = tmp .* (1 - resp{i}(:,nonzero_grad).^2);
    
      if (i-1) >= 1
	if (size(W{i}, 2) == size(resp{i-1}, 1) + 1)
	  grad{i} = backward{i} * [resp{i-1}(:,nonzero_grad)' ones(snonzero_grad, 1)];
	else
	  grad{i} = backward{i} * resp{i-1}(:,nonzero_grad)';
	end    
      else
	grad{i} = backward{i} * X(:, nonzero_grad)';
      end
    end
  else
    for i=nlayer:-1:1
      if (i == nlayer)
	tmp = hout;
      else
	tmp = (W{i+1}' * backward{i+1});
      end
      
      if (size(tmp, 1) ~= size(resp{i}, 1))
	tmp = tmp(1:end-1, :);
      end
      
      backward{i} = tmp .* (1 - resp{i}.^2);
      
      if (i-1) >= 1
	if (size(W{i}, 2) == size(resp{i-1}, 1) + 1)
	  grad{i} = backward{i} * [resp{i-1}' onesncases];
	else
	  grad{i} = backward{i} * resp{i-1}';
	end
      else
	grad{i} = backward{i} * X';
      end
    end
  end
  
elseif (nonlinearity == 2)

  if (exist('nonzero_grad', 'var'))
    snonzero_grad = sum(nonzero_grad);
    if (snonzero_grad == 0)
      for i=nlayer:-1:1
	grad{i} = zeros(size(W{i}));
      end
      return;
    end
    
    for i=nlayer:-1:1
      if (i == nlayer)
	tmp = hout(:, nonzero_grad);
      else
	tmp = (W{i+1}' * backward{i+1});
      end
      
      if (size(tmp, 1) ~= size(resp{i}, 1))
	tmp = tmp(1:end-1, :);
      end
      
      backward{i} = tmp .* (1 - resp{i}(:,nonzero_grad)) .* resp{i}(:,nonzero_grad);
      
      if (i-1) >= 1
	if (size(W{i}, 2) == size(resp{i-1}, 1) + 1)
	  grad{i} = backward{i} * [resp{i-1}(:,nonzero_grad)' gones(snonzero_grad,1,gputype)];
	else
	  grad{i} = backward{i} * resp{i-1}(:,nonzero_grad)';
	end
      else
	grad{i} = backward{i} * X(:, nonzero_grad)';
      end
    end
  else
    for i=nlayer:-1:1
      if (i == nlayer)
	tmp = hout;
      else
	tmp = (W{i+1}' * backward{i+1});
      end
      
      if (size(tmp, 1) ~= size(resp{i}, 1))
	tmp = tmp(1:end-1, :);
      end
      
      backward{i} = tmp .* (1 - resp{i}) .* resp{i};
      
      if (i-1) >= 1
	if (size(W{i}, 2) == size(resp{i-1}, 1) + 1)
	  grad{i} = backward{i} * [resp{i-1}' onesncases];
	else
	  grad{i} = backward{i} * resp{i-1}';
	end
      else
	grad{i} = backward{i} * X';
      end
    end
  end
  
elseif (nonlinearity == 3)
  
  if (exist('nonzero_grad', 'var'))
    snonzero_grad = sum(nonzero_grad);
    if (snonzero_grad == 0)
      for i=nlayer:-1:1
	grad{i} = zeros(size(W{i}));
      end
      return;
    end
  
    for i=nlayer:-1:1
      if (i == nlayer)
	tmp = hout(:, nonzero_grad);
      else
	tmp = (W{i+1}' * backward{i+1});
      end
    
      if (size(tmp, 1) ~= size(resp{i}, 1))
	tmp = tmp(1:end-1, :);
      end
      
      if (i == nlayer)
	backward{i} = tmp .* (1 - resp{i}(:,nonzero_grad)^2);
      else
	backward{i} = tmp .* (1 - resp{i}(:,nonzero_grad)) .* resp{i}(:,nonzero_grad);
      end
      
      if (i-1) >= 1
	if (size(W{i}, 2) == size(resp{i-1}, 1) + 1)
	  grad{i} = backward{i} * [resp{i-1}(:,nonzero_grad)' gones(snonzero_grad,1,gputype)];
	else
	  grad{i} = backward{i} * resp{i-1}(:,nonzero_grad)';
	end
      else
	grad{i} = backward{i} * X(:, nonzero_grad)';
      end
    end
  else
    for i=nlayer:-1:1
      if (i == nlayer)
	tmp = hout;
      else
	tmp = (W{i+1}' * backward{i+1});
      end
      
      if (size(tmp, 1) ~= size(resp{i}, 1))
	tmp = tmp(1:end-1, :);
      end
    
      if (i == nlayer)
	backward{i} = tmp .* (1 - resp{i}.^2);
      else
	backward{i} = tmp .* (1 - resp{i}) .* resp{i};
      end

      if (i-1) >= 1
	if (size(W{i}, 2) == size(resp{i-1}, 1) + 1)
	  grad{i} = backward{i} * [resp{i-1}' onesncases];
	else
	  grad{i} = backward{i} * resp{i-1}';
	end
      else
	grad{i} = backward{i} * X';
      end
    end
  end

end
