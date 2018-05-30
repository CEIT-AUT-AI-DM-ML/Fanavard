X=[a0(1:10998),a0(2:10999)]

numMFs = 10;
mfType = 'gbellmf';
in_fis = genfis1(X,numMFs,mfType);
epoch_n = 20;

dispOpt = zeros(1,4);
out_fis = anfis(X,in_fis,20,dispOpt);


evalfis(X(:,1),out_fis)
plot (evalfis(X(:,1),out_fis))
hold on
plot (X(:,2))