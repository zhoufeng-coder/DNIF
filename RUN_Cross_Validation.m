clear   % clear workspace
clc     % clear console screen
diary off;  diary on;   % to save console output
% The location of the folder that contains the data
path='data\';
datasets={'gg'};
classifier='rls_wnn';
%WKNKN
use_WKNKN = 1;      % 1=yes, 0=no
K = 1;              % number of K nearest known neighbors
eta = 0.1;          % decay rate 
if strcmp(classifier,'rls_wnn')
    use_W_matrix = 0;
end
% CROSS VALIDATION SETTING -----------------
cv_setting = 'cv_p';    % PAIR PREDICTION CASE
% CROSS VALIDATION PARAMETERS --------------
m = 100;  % number of n-fold experiments (repetitions)
n = 5; % the 'n' in "n-fold experiment"
disp('==============================================================');
fprintf('\nClassifier Used: %s',classifier);
switch cv_setting
    case 'cv_p', fprintf('\nCV Setting Used: CV_p - Pair Prediction\n');
end
if use_WKNKN
    fprintf('\nusing WKNKN: K=%i, eta=%g\n',K,eta);
end
fprintf('\n');
for ds=[1]
    disp('--------------------------------------------------------------');

    fprintf('\nData Set: %s\n', datasets{ds});

    % LOAD DATA
    [Y,Sd,St,Did,Tid]=getdata(path,datasets{ds});

    % PREDICT (+ print evaluation metrics)
    crossValidation(Y',Sd,St,classifier,cv_setting,m,n,use_WKNKN,K,eta,use_W_matrix);

    disp('--------------------------------------------------------------');
    diary off;  diary on;
end
disp('==============================================================');
diary off;