
function [K,eta]=alg_rls_parest(Y,Sd,St,cv_setting,nr_fold,left_out,use_WKNKN,K,eta,use_W_matrix)


    %--------------------------------------------------------------------

    % ranges of parameter values to be tested to identify best combination

    range_K          = [  1 2 3 4 5 6 7 8 9];

    range_eta        = [0.1   0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];

    % stopping criterion (number of iterations)
    num_iter = 2;

    %--------------------------------------------------------------------

    test_ind = get_test_indices(Y,cv_setting,left_out); % indices of the test set samples
    folds = get_folds(Y,cv_setting,nr_fold,left_out);   % folds of the CV done on the training set

    %--------------------------------------------------------------------

    y2s    = cell(1,nr_fold);
    Ws     = cell(1,nr_fold);
%     initAs = cell(length(range_k),nr_fold);
%     initBs = cell(length(range_k),nr_fold);
    for i=1:nr_fold
        y2 = Y;
        y2(folds{i}) = 0;  % folds{i} is the validation set
%         if use_WKNKN
%             y2 = preprocess_WKNKN(y2,Sd,St,K,eta);   % preprocessing Y
%         end
        y2s{i} = y2;
        
%         for k=1:length(range_k)
%             [initAs{k,i},initBs{k,i}] = initializer(y2,range_k(k));
%         end

        W = ones(size(y2));
        W(test_ind) = 0;
        W(folds{i}) = 0;
        Ws{i} = W;
    end

    %--------------------------------------------------------------------

    best_AUC = -Inf;

     for K =range_K 
         for eta =range_eta
                    % get overall AUPR for current parameter combination
                    AUCs = zeros(nr_fold,1);
                    for i=1:nr_fold
                        y2 = y2s{i};        %

                        y3 = alg_rls_predict(y2,Sd,St,cv_setting,nr_fold,left_out,use_WKNKN,K,eta,use_W_matrix);


                        [AUCs(i),~] = returnEvaluationMetrics(Y(folds{i}),y3(folds{i}));   % EVALUATE
                    end
                    auc_res = mean(AUCs);

                    if best_AUC < auc_res
                        best_AUC = auc_res;

                        best_K = K;
                        best_eta = eta;

                    end
                    

        end
    end

    %--------------------------------------------------------------------

    % return best parameters

           K = best_K;
           eta = best_eta;
%     lambda_d = best_lambda_d;
%     lambda_t = best_lambda_t;

    %--------------------------------------------------------------------

end