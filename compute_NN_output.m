function resp = compute_NN_output(W, X, nonlinearity)

% nonlinearity = 1 -- hyperbolic tangent
% nonlinearity = 2 -- logistic sigmoid (-0.5 last layer)
% nonlinearity = 3 -- logistic sigmoid + hyperbolic tangent (last layer)

if isa(X, 'gsingle')
  gputype = 'gsingle';
elseif isa(X, 'gdouble')
  gputype = 'gdouble';
end
nlayer = numel(W);

if (abs(nonlinearity) == 1)
  
  for i=1:nlayer
    if (i-1 == 0)
      last_resp = X;
    else
      last_resp = resp{i-1};
    end
    if (i>1)
      if isa(X, 'gsingle') || isa(X, 'gdouble')
	resp{i} = tanh( W{i} * [last_resp; gones(1,size(last_resp,2), gputype)] );
      else
	resp{i} = tanh( W{i} * [last_resp; ones(1,size(last_resp,2))] );
      end
    else
      resp{i} = tanh(W{i} * last_resp);
    end
  end
  
elseif (abs(nonlinearity) == 2)
  
  for i=1:nlayer
    if (i-1 == 0)
      last_resp = X;
    else
      last_resp = resp{i-1};
    end

    if (i>1)
      if isa(X, 'gsingle') || isa(X, 'gdouble')
	resp{i} = 1./(1+exp(-W{i} * [last_resp; gones(1,size(last_resp,2),gputype)]));
      else
	resp{i} = 1./(1+exp(-W{i} * [last_resp; ones(1,size(last_resp,2))]));
      end
    else
      resp{i} = 1./(1+exp(-W{i} * last_resp));
    end
  end
  resp{end} = resp{end} - .5;
  
elseif (abs(nonlinearity) == 3)

  for i=1:nlayer
    if (i-1 == 0)
      last_resp = X;
    else
      last_resp = resp{i-1};
    end
    if (i>1)
      if (i == nlayer)
	if isa(X, 'gsingle') || isa(X, 'gdouble')
	  resp{i} = tanh( W{i} * [last_resp; gones(1,size(last_resp,2), gputype)] );
	else
	  resp{i} = tanh( W{i} * [last_resp; ones(1,size(last_resp,2))] );
	end
      else
	if isa(X, 'gsingle') || isa(X, 'gdouble')
	  resp{i} = 1./(1+exp(-W{i} * [last_resp; gones(1,size(last_resp,2),gputype)]));
	else
	  resp{i} = 1./(1+exp(-W{i} * [last_resp; ones(1,size(last_resp,2))]));
	end
      end
    else
      resp{i} = 1./(1+exp(-W{i} * last_resp));
    end
  end

end

if (nonlinearity < 0)
  i = nlayer;
  
  if (i>1)
    if isa(X, 'gsingle') || isa(X, 'gdouble')
      resp{i} = W{i} * [resp{i-1}; gones(1,size(resp{i-1},2),gputype)];
    else
      resp{i} = W{i} * [resp{i-1}; ones(1,size(resp{i-1},2))];
    end
  else
    resp{i} = W{i} * resp{i-1};
  end
end
