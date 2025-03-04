close all;
clear;
clc;

preprocessing_method = 2;   %select preprocessing_method = 2 for standardization to zero mean_unit variance
Ratio = [0.8 0.2 0];
select_kfolds = 5; 			%5-fold cross validation selection
total_features = [2 8 16];   
total_cluster_radius = [0.3 0.6 0.9];


%import data from datasheet
data = importdata('Superconductivity.csv');
colHeaders = data.colheaders;
data = data.data;

% Split the data with standardization preprocessing method
[training_data, validation_data, ~] = split_scale(data, preprocessing_method, Ratio);

% Grid Search MSE results and Number of rules for each model
MSE_grid_search = NaN * ones(length(total_features),length(total_cluster_radius));
model_rules = NaN * ones(length(total_features),length(total_cluster_radius));

% Use Relief algorithm to find the most important features
[Indices,Weights] = relieff(training_data(:,1:end-1),training_data(:,end),10);

% Perform a grid search on the parameter total_features and total_cluster_radius
for current_features_number = 1:length(total_features)
    current_training_data = training_data(:,Indices(1:total_features(current_features_number)));
    current_training_data = [current_training_data training_data(:,end)];
	
    for current_cluster_radius = 1:length(total_cluster_radius)
        clc;  
        fprintf('Current state is: (%d,%d) out of (%d,%d) grid', current_features_number,current_cluster_radius,length(total_features),length(total_cluster_radius));        
        cross_validation = cvpartition(size(training_data,1),'KFold',select_kfolds);
        % MSE matrix for k-folds cross validation
        mse_matrix = NaN * ones(1, select_kfolds);

        for current_kfolds = 1:select_kfolds
            % Split the training data into training and validation
            cv_training_data = current_training_data(training(cross_validation,current_kfolds),:);
            cv_validation_data = current_training_data(test(cross_validation,current_kfolds),:);
            
            % Model training and performance measurment
            Options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', total_cluster_radius(current_cluster_radius));
            fis = genfis(cv_training_data(:,1:end-1), cv_training_data(:,end), Options);
            
            % If fis rules < 2 skip to the next iteration
            if (size(fis.Rules, 2) < 2)
                continue;
            end
                        
            [fis_training, training_RMSE, step, best_validation, RMSE_validation] = anfis(cv_training_data,anfisOptions('InitialFis', fis, 'EpochNumber', 100, 'ValidationData', cv_validation_data, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0,'DisplayStepSize',0, 'DisplayFinalResults',0));
            
            % Store the MSE as this model's performance metric
            mse_matrix(current_kfolds) = min(RMSE_validation.*RMSE_validation);
        end
        MSE_grid_search(current_features_number,current_cluster_radius) = mean(mse_matrix); % Store the mean MSE from the %k_folds iterations of CVs
        model_rules(current_features_number,current_cluster_radius) = size(fis.Rules, 2);   % Store the number of rules
    end
end

% Error - number of rules plot
figure('Position',[75 70 1400 680]);
title('Error - Number of Rules');
subplot(1,2,1), scatter(reshape(model_rules,1,[]), reshape(MSE_grid_search,1,[])), hold on;
ylabel('MSE');
xlabel('Number of Rules');

temp = 0;
colors = {'blue','green','red','black'};
for i = min(reshape(model_rules,1,[])): max(reshape(model_rules,1,[]))
    xline(i,colors{mod(temp, 4) + 1});
    temp = temp + 1;
end

subplot(1,2,2), boxplot(MSE_grid_search',total_features);
title('Error - Number of Features');
ylabel('MSE');
xlabel('Number of Features');

%------------------ Build the optimal model----------------------

% Find the best parameters
[~,best_MSE_idx] = min(MSE_grid_search,[],'all','omitnan','linear');
[best_MSE_row,best_MSE_column] = ind2sub(size(MSE_grid_search),best_MSE_idx);
best_features_number = total_features(best_MSE_row);
best_cluster_radius = total_cluster_radius(best_MSE_column);

%Find best training and validation data
best_training_data = training_data(:,Indices(1:best_features_number));
best_validation_data = validation_data(:,Indices(1:best_features_number));
best_training_data = [best_training_data training_data(:,end)];
best_validation_data = [best_validation_data validation_data(:,end)];

%Split data again for validation data
new_ratio = [0.75 0.25 0];
new_preprocessing_method = 0;
[best_training_data, best_val_data] = split_scale(best_training_data, new_preprocessing_method, new_ratio);

% Model training and performance measurment
best_fis = genfis(best_training_data(:,1:end-1), best_training_data(:,end),genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', best_cluster_radius));
[best_fis_training, best_training_RMSE,~, fis_best_validation, best_RMSE_validation] = anfis(best_training_data, anfisOptions('InitialFis',best_fis,'EpochNumber', 100, 'ValidationData',best_val_data,'DisplayANFISInformation',0, 'DisplayErrorValues',1,'DisplayStepSize',0,'DisplayFinalResults',1));

% Best model parameters-metrics
R_squared = @(predicted_Y, actual_Y) 1 - sum((predicted_Y - actual_Y).^ 2)/sum((actual_Y - mean(actual_Y)).^ 2);
Predictions = evalfis(fis_best_validation,best_validation_data(:,1:end-1));
best_MSE = mse(Predictions,best_validation_data(:,end));
best_RMSE = sqrt(best_MSE);
best_R_squared = R_squared(Predictions,best_validation_data(:,end));
best_NMSE = 1 - best_R_squared;
best_NDEI = sqrt(best_NMSE);
best_performance_matrix = array2table([best_RMSE best_NMSE best_NDEI best_R_squared],'VariableNames',{'RMSE' 'NMSE' 'NDEI' 'R2'});
disp(best_performance_matrix)

% Predictions - actual values plot
plotregression(best_validation_data(:,end),Predictions);
title('Predictions - Actual Values');
ylabel('Predictions');
xlabel('Actual Values');
saveas(gcf, 'actual_values_predictions.png');

% Prediction error histogram
figure();
title('Best model prediction error percentage');
prediction_error = Predictions - best_validation_data(:,end);
prediction_error_percentage = prediction_error./best_validation_data(:,end);
histogram(prediction_error_percentage,round(length(prediction_error_percentage) * 15));
xlim([-3 10]);
ylabel('Frequency');
xlabel('Prediction Error (%)');
saveas(gcf, 'histogram_prediction_error.png');

% Learning curve plot
figure();
plot(best_RMSE,'red');
title('Best model Learning Curve');
hold on;
plot(best_RMSE_validation,'green');
legend('Training','Validation','Location','Best');
ylabel('RMSE');
xlabel('Iterations');
saveas(gcf,'best_model_learning_curve.png');

% models membership functions
figure('Position',[25 50 1500 730]);
subplot(2,3,1),plotmf(best_fis,'input',4);
subplot(2,3,2),plotmf(best_fis,'input',6);
subplot(2,3,3),plotmf(best_fis,'input',9);
subplot(2,3,4),plotmf(fis_best_validation,'input',4);
subplot(2,3,5),plotmf(fis_best_validation,'input',6);
subplot(2,3,6),plotmf(fis_best_validation,'input',9);
saveas(gcf,'models_mfs.png');