function [training_data,validation_data,testing_data] = split_scale(data, preprocessing_method, Ratio)
    %This function is responsible for splitting and preprocessing data
    idX = randperm(length(data));
    
    Train_idX = idX(1:round(length(idX) * Ratio(1)));
    Validate_idX = idX(round(length(idX) * Ratio(1))+1:round(length(idX) * (Ratio(1) + Ratio(2))));
    Test_idX = idX(round(length(idX) * (Ratio(1) + Ratio(2)))+1:end);
    
    X_Train = data(Train_idX,1:end-1);
    X_Validate = data(Validate_idX,1:end-1);
    X_Test = data(Test_idX,1:end-1);
    
    if preprocessing_method == 0
        %Do Nothing. preprocessing
    elseif preprocessing_method == 1  %Normalization to unit hypercube
        minX = min(X_Train,[],1);
        maxX = max(X_Train,[],1);
        X_Train = (X_Train-repmat(minX,[length(X_Train) 1]))./(repmat(maxX,[length(X_Train) 1])-repmat(minX,[length(X_Train) 1]));
        X_Validate = (X_Validate-repmat(minX,[length(X_Validate) 1]))./(repmat(maxX,[length(X_Validate) 1])-repmat(minX,[length(X_Validate) 1]));
        X_Test = (X_Test-repmat(minX,[length(X_Test) 1]))./(repmat(maxX,[length(X_Test) 1])-repmat(minX,[length(X_Test) 1]));
    elseif preprocessing_method == 2  %Standardization to zero mean_unit variance
        mean_unit = mean(data(:,1:end-1));
        sig = std(data(:,1:end-1));
        X_Train = (X_Train-repmat(mean_unit,[length(X_Train) 1]))./repmat(sig,[length(X_Train) 1]);
        X_Validate = (X_Validate-repmat(mean_unit,[length(X_Validate) 1]))./repmat(sig,[length(X_Validate) 1]);
        X_Test = (X_Test-repmat(mean_unit,[length(X_Test) 1]))./repmat(sig,[length(X_Test) 1]);
    else
        disp('Error: Preprocessing Method id is not valid!')
    end
    
    training_data = [X_Train data(Train_idX,end)];
    validation_data = [X_Validate data(Validate_idX,end)];
    testing_data = [X_Test data(Test_idX,end)];
end