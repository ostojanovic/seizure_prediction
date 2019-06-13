function [model,parameter]= fit_polynominal(data,order)

x_temp = 1:size(data,1);
x_temp = x_temp-mean(x_temp);
x_temp = (x_temp./max(x_temp))';                                      % centered around zero; between -1 and 1
x_vec = repmat(x_temp,1,order);

for IDXO = 1:order
    x_vec(:,IDXO) = x_vec(:,IDXO).^IDXO;
end

for IDXE = 1:size(data,2)
    parameter(IDXE,:) = robustfit(x_vec,data(:,IDXE));
    model(IDXE,:)     = parameter(IDXE,1)+parameter(IDXE,2:end)*x_vec';
end