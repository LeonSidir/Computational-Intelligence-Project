clear;
clc
close all
format compact
% Import and split data (split_scale)
preprocessing_method = 2; %select preprocessing_method = 1 for standardization to zero mean_unit variance
performance_matrix = zeros(4,4); %Initialize models' performance matrix with zeros
data = load('airfoil_self_noise.dat'); %import the data
[Rows, Columns] = size(data);
FeaturesNumber = Columns - 1;
Ratio = [0.6 0.2 0.2];
[training_data,validation_data,testing_data] = split_scale(data,preprocessing_method, Ratio); %split the data with preprocessing_method = 1

% FIS(Fuzzy Inference System) with grid partition
Models = [genfis1(training_data, 2,'gbellmf','constant'), genfis1(training_data, 3,'gbellmf','constant'), genfis1(training_data, 2,'gbellmf','linear'), genfis1(training_data, 3,'gbellmf','linear')];    
% R2
R_squared = @(y_predicted, y_actual) 1 - sum((y_predicted - y_actual).^2) / sum((y_actual - mean(y_actual)).^2);

for i = 1:4
    % Figure for membership functions of the features
    [Fis_training,Error_training,~,Fis_validation,Error_validation] = anfis(training_data, Models(i), [100 0 0.01 0.9 1.1], [], validation_data);
    for j = 1:5 
        figure();
        grid on
        title("TSK_Model" + i + "Feature "+ j);
        legend('Training','Validation', 'Location', 'east');
        plotmf(Fis_validation,'input',j);
        saveas(gcf,"TSK_Model_" + i + "_Feature" + j +".png")
    end
    
    % Learning curve plots
    figure();
    grid on
    plot([Error_training Error_validation]);
    legend('Training','Validation');
    ylabel('Error')
    xlabel('Iterations');
    title("TSK Model " + i + " Learning Curve");
    saveas(gcf,"TSK_Model_" + i + "_learning_curve.png")
    
    % Performance Parameters
    y = evalfis(testing_data(:,1:end-1),Fis_validation);
    R2 = R_squared(y,testing_data(:,end));
    RMSE = sqrt(mse(y,testing_data(:,end)));
    NMSE = 1 - R2;
    NDEI = sqrt(NMSE);
    performance_matrix(:,i) = [R2; RMSE; NMSE; NDEI];
    
    % Prediction error plot
    predict_error = testing_data(:,end) - y; 
    figure();
    plot(predict_error);
    title("TSK Model " + i + " Prediction Error ");
    grid on;
    ylabel('Error');
    xlabel('Input');
    saveas(gcf, "TSK_Model_"+ i + "_prediction_error.png")
end

% Store the results
Row_Names={'R2' , 'RMSE' , 'NMSE' , 'NDEI'};
Variable_Names={'TSK_Model_1', 'TSK_Model_2', 'TSK_Model_3', 'TSK_Model_4'};
performance_matrix = array2table(performance_matrix,'VariableNames',Variable_Names,'RowNames',Row_Names);
disp(performance_matrix)