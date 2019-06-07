
time_vec = 0.1:0.1:20;
y_vec    = 0.1:0.2:4;

time_factor_1 = time_vec.^2;
time_factor_2 = time_vec.^3;
time_factor_2 = max(time_factor_2)-time_factor_2;
time_factor_2 = time_factor_2./max(time_factor_2);
time_factor_1 = time_factor_1./max(time_factor_1);

y_factor_1 = (y_vec -mean(y_vec )).^2;
y_factor_2 = (y_vec -mean(y_vec )).^3;
y_factor_2 = max(y_factor_2)-y_factor_2;
y_factor_2 = y_factor_2./max(y_factor_2);
y_factor_1 = y_factor_1./max(y_factor_1);

Mat1   = time_factor_1' *y_factor_1;
Mat2   = time_factor_1' *y_factor_2;
Mat3   = (time_factor_2' *y_factor_1).^8;
Mat4   = time_factor_2' *y_factor_2;

[W,H] = nnmf(Mat1,1);
[W,H] = nnmf(Mat2,1);
[W,H] = nnmf(Mat3,1);
[W,H] = nnmf(Mat4,1);

[W,H] = nnmf(Mat4+Mat1,2);
temp = W*H;

[W,H] = nnmf(Mat4+Mat1,1);
temp = W*H;

noise_mat = 0.1.*rand(size(Mat1));

[W,H] = nnmf((Mat4+Mat1)+noise_mat,2);
temp = W*H;

[W,H] = nnmf(Mat4+Mat1+noise_mat,1);
temp = W*H;

%%% use now the result for robust fit
% start with one dimension 2 matrices W (Yx2)
y = W(:,1);

x_vec = 1:size(W,1);
x_vec = x_vec-mean(x_vec);
x_vec = x_vec./max(x_vec);
DesignMat  = repmat(x_vec,3,1);
DesignMat(2,:) = DesignMat(2,:).^2;
DesignMat(3,:) = DesignMat(3,:).^3;
b = robustfit(DesignMat',y);

y_predict = b(2:end)'*DesignMat+b(1);
plot(y_predict)
