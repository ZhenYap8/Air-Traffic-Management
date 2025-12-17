%% EMS740 - Machine Learning & AI in Engineering
% Case Study: Artificial Intelligence in Air Traffic Management
% Task: ANFIS Modelling for Taxi Time Prediction

%% 1. Data Import
% We use 'NumHeaderLines', 1 to safely skip the text headers in the CSVs
% effectively preventing NaN errors during import.
fprintf('>> Loading datasets...\n');

raw_train = readmatrix('PCA_train_reduced_gX.csv', 'NumHeaderLines', 1); 
raw_val   = readmatrix('PCA_validation_reduced_gX.csv', 'NumHeaderLines', 1);
raw_test  = readmatrix('PCA_test_reduced_gX.csv', 'NumHeaderLines', 1);

% Split into Inputs (Features) and Targets (Taxi Time)
% The target variable is in the last column.
X_train = raw_train(:, 1:end-1);
Y_train = raw_train(:, end);

X_val = raw_val(:, 1:end-1);
Y_val = raw_val(:, end);

X_test = raw_test(:, 1:end-1);
Y_test = raw_test(:, end);

%% 2. Feature Standardization
% Standardizing to zero mean and unit variance is crucial for ANFIS performance.
% IMPORTANT: We calculate mean/std from the TRAINING set only to prevent 
% data leakage (cheating) into the validation/test sets.

mu = mean(X_train);
sigma = std(X_train);

% Apply the transformation
X_train_std = (X_train - mu) ./ sigma;
X_val_std   = (X_val - mu) ./ sigma;
X_test_std  = (X_test - mu) ./ sigma;

% Combine back into one matrix for the 'anfis' function
data_train = [X_train_std, Y_train];
data_val   = [X_val_std, Y_val];
data_test  = [X_test_std, Y_test];

%% 3. FIS Structure Generation (Clustering)
% We selected Subtractive Clustering over Grid Partitioning to avoid the 
% "curse of dimensionality" and massive rule bases.
%
% Sensitivity Analysis: 
% We tested radii of 0.25, 0.3, and 0.5. 
% Radius 0.3 provided the best balance between overfitting and underfitting.

radius = 0.3; 
fprintf('>> Generating initial FIS with Cluster Radius: %.2f\n', radius);

opt_gen = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', radius);
initial_fis = genfis(X_train_std, Y_train, opt_gen);

%% 4. Model Training
% We use the Hybrid algorithm (Backprop + Least Squares) for faster convergence.
% Validation data is used to stop training early if overfitting occurs.

opt_train = anfisOptions('InitialFIS', initial_fis);
opt_train.ValidationData = data_val;
opt_train.EpochNumber = 100;
opt_train.OptimizationMethod = 1; % 1 = Hybrid method
opt_train.DisplayANFISInformation = 0; % Keep command window clean

fprintf('>> Training ANFIS model... (Please wait)\n');
[fis_final, trainError, stepSize, fis_best, valError] = anfis(data_train, opt_train);

% 'fis_best' is the version of the model that had the lowest validation error.

%% 5. Performance Evaluation
fprintf('>> Testing model on unseen data...\n');

% Predict using the best model found during training
Y_pred = evalfis(fis_best, X_test_std);

% Calculate key metrics
rmse = sqrt(mean((Y_pred - Y_test).^2));
mae  = mean(abs(Y_pred - Y_test));
% R-Squared calculation
SStot = sum((Y_test - mean(Y_test)).^2);
SSres = sum((Y_test - Y_pred).^2);
R2 = 1 - SSres / SStot;

% Print concise results to console
fprintf('\n----------------------------------\n');
fprintf(' FINAL RESULTS (Group X)\n');
fprintf('----------------------------------\n');
fprintf(' Cluster Radius: %.2f\n', radius);
fprintf(' RMSE:           %.4f min\n', rmse);
fprintf(' MAE:            %.4f min\n', mae);
fprintf(' R-Squared:      %.4f\n', R2);
fprintf('----------------------------------\n');

%% 6. Statistical Bias Check
% We check if the model consistently over/under-predicts.
% We calculate the 95% Confidence Interval (CI) of the mean error.

residuals = Y_pred - Y_test;
mean_err = mean(residuals);
std_err = std(residuals);

% Calculate 95% CI
alpha = 0.05;
n_samples = length(residuals);
t_score = tinv([0.025  0.975], n_samples-1); 
SEM = std_err / sqrt(n_samples); % Standard Error
CI = mean_err + t_score * SEM; 

fprintf('\n>> Statistical Bias Analysis:\n');
fprintf(' Mean Error: %.4f\n', mean_err);
fprintf(' 95%% CI:    [%.4f, %.4f]\n', CI(1), CI(2));

if CI(1) <= 0 && CI(2) >= 0
    fprintf(' Conclusion: Model is UNBIASED (Zero is inside the CI).\n');
else
    fprintf(' Conclusion: Significant BIAS detected.\n');
end
fprintf('----------------------------------\n');

%% 7. Visualization: Performance
% Plot 1: Learning Curve & Scatter Plot
fig1 = figure('Name', 'ANFIS Performance', 'NumberTitle', 'off', 'Position', [100, 100, 900, 700]);

subplot(2,1,1);
plot(trainError, 'b', 'LineWidth', 1.5); hold on;
plot(valError, 'r', 'LineWidth', 1.5);
legend('Training Error', 'Validation Error');
title(sprintf('Learning Curve (Radius %.2f)', radius));
xlabel('Epochs'); ylabel('RMSE'); grid on;

subplot(2,1,2);
scatter(Y_test, Y_pred, 30, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
plot([min(Y_test) max(Y_test)], [min(Y_test) max(Y_test)], 'r', 'LineWidth', 2); % 1:1 Line
title(sprintf('Actual vs Predicted Taxi Time (R^2 = %.3f)', R2));
xlabel('Actual Time (min)'); ylabel('Predicted Time (min)'); grid on;

saveas(fig1, 'ANFIS_Performance_Plot.png');
fprintf('>> Saved plot: ANFIS_Performance_Plot.png\n');

%% 8. Visualization: Statistical Proof
% Plot 2: Residuals with Confidence Intervals
fig2 = figure('Name', 'Statistical Proof', 'NumberTitle', 'off', 'Position', [150, 150, 900, 700]);

subplot(2,1,1);
plot(residuals, 'b.', 'MarkerSize', 5); hold on;
yline(0, 'k-', 'LineWidth', 1.5); 
yline(CI(1), 'r--', 'LineWidth', 1.5, 'Label', 'Lower 95%');
yline(CI(2), 'r--', 'LineWidth', 1.5, 'Label', 'Upper 95%');
yline(mean_err, 'g-', 'LineWidth', 1.5, 'Label', 'Mean Error');
title('Residual Analysis with 95% Confidence Interval');
ylabel('Error (min)'); xlabel('Test Samples');
ylim([-15 15]); grid on;
legend('Residuals', 'Zero Line', '95% CI', '', 'Mean Error');

subplot(2,1,2);
histogram(residuals, 40, 'FaceColor', [0.6 0.6 0.6]); hold on;
xline(0, 'k-', 'LineWidth', 2);
xline(CI(1), 'r--', 'LineWidth', 2);
xline(CI(2), 'r--', 'LineWidth', 2);
title('Error Distribution Histogram');
xlabel('Prediction Error (min)'); ylabel('Count');
legend('Distribution', 'Zero', '95% CI'); grid on;

saveas(fig2, 'ANFIS_Confidence_Analysis.png');
fprintf('>> Saved plot: ANFIS_Confidence_Analysis.png\n');

%% 9. Save Results to File
% Export metrics for the group report
outfile = 'ANFIS_Final_Results.txt';
fid = fopen(outfile, 'w');

fprintf(fid, 'ANFIS Model Results - Group X\n');
fprintf(fid, '=============================\n');
fprintf(fid, 'Cluster Radius: %.2f\n', radius);
fprintf(fid, 'RMSE:           %.4f\n', rmse);
fprintf(fid, 'MAE:            %.4f\n', mae);
fprintf(fid, 'R-Squared:      %.4f\n\n', R2);
fprintf(fid, 'Statistical Analysis\n');
fprintf(fid, '--------------------\n');
fprintf(fid, 'Mean Error:     %.4f\n', mean_err);
fprintf(fid, '95%% CI Lower:   %.4f\n', CI(1));
fprintf(fid, '95%% CI Upper:   %.4f\n', CI(2));

if CI(1) <= 0 && CI(2) >= 0
    fprintf(fid, 'Result:         UNBIASED\n');
else
    fprintf(fid, 'Result:         BIASED\n');
end

fclose(fid);
fprintf('>> All results exported to %s\n', outfile);
fprintf('>> Job Complete.\n');
%% 10. Model Transparency (Rule Extraction)
% This step extracts and displays the fuzzy logic rules learned by the model.
% We save them to a text file so you can include examples in your report.
% This proves the model is a "Grey Box" (interpretable) rather than a Black Box.

fprintf('\n>> Extracting Fuzzy Rules...\n');

% Get the rules from the best trained model
rules = showrule(fis_best);

% Display the first 5 rules in the Command Window as a preview
fprintf('---------------------------------------------------\n');
fprintf(' Model Transparency Check: First 5 Rules (Preview)\n');
fprintf('---------------------------------------------------\n');
disp(rules(1:5, :)); 
fprintf('... (Total rules: %d)\n', size(rules, 1));

% Save ALL rules to a text file for the report
ruleFile = 'ANFIS_Rules.txt';
fid_r = fopen(ruleFile, 'w');
fprintf(fid_r, 'ANFIS Learned Rules (Group X)\n');
fprintf(fid_r, '=============================\n');
fprintf(fid_r, 'Total Rules Generated: %d\n\n', size(rules, 1));

% Iterate and write each rule clearly
for i = 1:size(rules, 1)
    fprintf(fid_r, '%s\n', rules(i, :));
end

fclose(fid_r);
fprintf('>> All rules saved to %s. Include 2-3 examples in your report!\n', ruleFile);
fprintf('>> Job Complete.\n');