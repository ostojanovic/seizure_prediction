function [model,parameter]= fit_splines(data,order,lin_flag)

if lin_flag==1
    [Design_Matrix] = Create_splines_linspace(size(data,1), order, 0);
else
    [Design_Matrix] = Create_splines_logspace(size(data,1), order, 0);
end

for IDXE = 1:size(data,2)
    parameter(IDXE,:) = robustfit(Design_Matrix,data(:,IDXE));
    model(IDXE,:)     = parameter(IDXE,1)+parameter(IDXE,2:end)*Design_Matrix';
end
