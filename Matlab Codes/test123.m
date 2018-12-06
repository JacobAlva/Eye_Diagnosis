clc
clear

load traintable.mat 

table=cell2mat(table)
[valh indh]=find(table==3)
[vald indd]=find(table==2)
[valg indg]=find(table==1)


health=table(valh,1:4)
t1=[ones(1,numel(valh)) zeros(1,numel(vald)) zeros(1,numel(valg))]
diabetic=table(vald,1:4)
t2=[zeros(1,numel(vald)) ones(1,numel(valh)) zeros(1,numel(valg))]
glaucoma=table(valg,1:4)
t3=[zeros(1,numel(vald)) zeros(1,numel(valg)) ones(1,numel(valh))]

input=[health;diabetic;glaucoma]'
target=[t1;t2;t3]

% N=input(1,:)
% 
% S1=5;   % numbe of hidden layers
% S2=3;   % number of output layers (= number of classes)
% 
% P=input;
% T=target;
% 
% [R,Q]=size(P); 
% epochs = 1000;      % number of iterations
% goal_err = 10e-5;    % goal error
% a=0.3;                        % define the range of random variables
% b=-0.3;
% W1=a + (b-a) *rand(S1,R);     % Weights between Input and Hidden Neurons
% W2=a + (b-a) *rand(S2,S1);    % Weights between Hidden and Output Neurons
% b1=a + (b-a) *rand(S1,1);     % Weights between Input and Hidden Neurons
% b2=a + (b-a) *rand(S2,1);     % Weights between Hidden and Output Neurons
% n1=W1*P;
% A1=logsig(n1);
% n2=W2*A1;
% A2=logsig(n2);
% e=A2-T;
% error =0.5* mean(mean(e.*e));    
% nntwarn off
% for  itr =1:epochs
%     if error <= goal_err 
%         break
%     else
%          for i=1:Q
%             df1=dlogsig(n1,A1(:,i));
%             df2=dlogsig(n2,A2(:,i));
%             s2 = -2*diag(df2) * e(:,i);			       
%             s1 = diag(df1)* W2'* s2;
%             W2 = W2-0.1*s2*A1(:,i)';
%             b2 = b2-0.1*s2;
%             W1 = W1-0.1*s1*P(:,i)';
%             b1 = b1-0.1*s1;
% 
%             A1(:,i)=logsig(W1*P(:,i),b1);
%             A2(:,i)=logsig(W2*A1(:,i),b2);
%          end
%             e = T - A2;
%             error =0.5*mean(mean(e.*e));
%             disp(sprintf('Iteration :%5d        mse :%12.6f%',itr,error));
%             mse(itr)=error;
%     end
% end
% 
% save W1.mat W1
% save W2.mat W2
% save A1.mat A1
% save A2.mat A2
% 
% 
% threshold=0.9;   % threshold of the system (higher threshold = more accuracy)
% 
% % training images result
% 
% %TrnOutput=real(A2)
% TrnOutput=real(A2>threshold)    
% 
% % applying test images to NN
% n1=W1*N;
% A1=logsig(n1);
% n2=W2*A1;
% A2test=logsig(n2);
% 
% % testing images result
% 
% %TstOutput=real(A2test)
% TstOutput=real(A2test>threshold)
