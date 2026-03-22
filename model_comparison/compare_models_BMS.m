% ----------- Import Log Model Evidence (LME) data-----------%
bic_approx_lme_FilePath = 'LMEs_BIC_approx.csv'; 
aic_approx_lme_FilePath = 'LMEs_AIC_approx.csv'; 
bic_approx_lme_ddm_FilePath = 'LMEs_BIC_approx_ddm.csv'; 
aic_approx_lme_ddm_FilePath = 'LMEs_AIC_approx_ddm.csv';

modelNames = {'RL', 'AI0','AI1', 'AI2', 'AI3'};

bic_approx_lmeTable = readtable(bic_approx_lme_FilePath);
aic_approx_lmeTable = readtable(aic_approx_lme_FilePath);
bic_approx_lme_ddmTable = readtable(bic_approx_lme_ddm_FilePath);
aic_approx_lme_ddmTable = readtable(aic_approx_lme_ddm_FilePath);

allColNames_lme = bic_approx_lmeTable.Properties.VariableNames;

%disp(modelNames'); % Transpose for better display

allColNames_lme_ddm = bic_approx_lme_ddmTable.Properties.VariableNames;

ddmModelNames = {
    'RL_ddm_DRMlinear', 
    'AI_ddm0_DRMlinear', 
    'AI_ddm1_DRMlinear', 
    'AI_ddm2_DRMlinear', 
    'AI_ddm3_DRMlinear', 
    %'RL_ddm_DRMsigmoid_single_v_mod', 
    %'AI_ddm0_DRMsigmoid_single_v_mod',
    %'AI_ddm1_DRMsigmoid_single_v_mod',
    %'AI_ddm2_DRMsigmoid_single_v_mod',
    %'AI_ddm3_DRMsigmoid_single_v_mod'
};

% ----------- Build LME arrays -----------%
bic_approx_lmeSubTable = bic_approx_lmeTable(:, modelNames);
BIC_approx_LME_matrix = table2array(bic_approx_lmeSubTable);

aic_approx_lmeSubTable = aic_approx_lmeTable(:, modelNames);
AIC_approx_LME_matrix = table2array(aic_approx_lmeSubTable);

bic_approx_lme_ddmSubTable = bic_approx_lme_ddmTable(:, ddmModelNames);
BIC_approx_LME_ddm_matrix = table2array(bic_approx_lme_ddmSubTable);

aic_approx_lme_ddmSubTable = aic_approx_lme_ddmTable(:, ddmModelNames);
AIC_approx_LME_ddm_matrix = table2array(aic_approx_lme_ddmSubTable);

% --- Define output CSV filenames ---

outputCsvFile_standard_BIC_approx = 'bms_results_bic_approx_standard_models.csv';
outputCsvFile_standard_AIC_approx = 'bms_results_aic_approx_standard_models.csv';
outputCsvFile_ddm_BIC_approx = 'bms_results_bic_approx_ddm_models.csv';
outputCsvFile_ddm_AIC_approx = 'bms_results_aic_approx_ddm_models.csv';
outputCsvFile_ddm_families_BIC_approx = 'bms_results_bic_approx_ddm_families.csv';
outputCsvFile_ddm_families_AIC_approx = 'bms_results_aic_approx_ddm_families.csv';

outputCsvFile_ddm_families_comparison_BIC_approx = 'bms_results_bic_approx_ddm_families_comparison.csv';
outputCsvFile_ddm_families_comparison_AIC_approx = 'bms_results_aic_approx_ddm_families_comparison.csv';

% --- Run Bayesian Standard Model Selection with BIC approx ---

fprintf('\n--- Running Bayesian Standard Model Selection with BIC approx---\n');

try
    [alpha, exp_r, xp, pxp, bor] = spm_BMS(BIC_approx_LME_matrix);

    [posterior, out] = VBA_groupBMC(BIC_approx_LME_matrix');

    disp('Exceedance Probabilities according to VBA toolbox:');
    disp(out.ep');

    %disp('Alpha (Dirichlet parameters for model posterior):');
    %disp(alpha'); % Transpose for easier reading if many models

    disp('Estimated Model Frequency:');
    disp(out.Ef);

    disp('Estimated Model Frequency Variance:');
    disp(out.Vf);

    disp('Expected Posterior Probability (exp_r) for each model:');
    for m = 1:numel(modelNames)
        fprintf('  %s: %.4f\n', modelNames{m}, exp_r(m));
    end

    disp('Exceedance Probability (xp) for each model:');
    for m = 1:numel(modelNames)
        fprintf('  %s: %.4f\n', modelNames{m}, xp(m));
    end

    disp('Protected Exceedance Probability (pxp) for each model:');
     for m = 1:numel(modelNames)
        fprintf('  %s: %.4f\n', modelNames{m}, pxp(m));
    end

    fprintf('Bayes Omnibus Risk (BOR - prob. that models are equally likely): %.4f\n', bor);

     % --- Save results to CSV for standard models ---
    % Ensure modelNames is a column vector for table creation
    if isrow(modelNames)
        modelNamesCol = modelNames';
    else
        modelNamesCol = modelNames;
    end
    % Ensure EP and PEP are column vectors
    exp_rCol = exp_r';
    xpCol = xp';
    pxpCol = pxp';
    emfCol = out.Ef;
    emfVarCol = out.Vf;

    resultsTable = table(modelNamesCol, exp_rCol, xpCol, pxpCol, emfCol, emfVarCol, ...
        'VariableNames', {'ModelName', ...
        'Expected Posterior Probability', ...
        'ExceedanceProbability', ...
        'ProtectedExceedanceProbability', ...
        'EstimatedModelFrequency', ...
        'EstimatedModelFrequencyVariance'});
    writetable(resultsTable, outputCsvFile_standard_BIC_approx);
    fprintf('Standard model BMS results with BIC approx saved to %s\n', ...
        outputCsvFile_standard_BIC_approx);

catch ME_bms_bic_approx
    fprintf('Error during BMS execution for standard models with BIC approx:\n');
    rethrow(ME_bms_bic_approx);
end

% --- Run Bayesian Standard Model Selection with AIC approx ---

fprintf('\n--- Running SPM Bayesian Standard Model Selection with AIC approx---\n');

try
    [alpha, exp_r, xp, pxp, bor] = spm_BMS(AIC_approx_LME_matrix);

    [posterior, out] = VBA_groupBMC(AIC_approx_LME_matrix');

    disp('Exceedance Probabilities according to VBA toolbox:');
    disp(out.ep');

    %disp('Alpha (Dirichlet parameters for model posterior):');
    %disp(alpha'); % Transpose for easier reading if many models

    disp('Estimated Model Frequency:');
    disp(out.Ef);

    disp('Estimated Model Frequency Variance:');
    disp(out.Vf);

    disp('Expected Posterior Probability (exp_r) for each model:');
    for m = 1:numel(modelNames)
        fprintf('  %s: %.4f\n', modelNames{m}, exp_r(m));
    end

    disp('Exceedance Probability (xp) for each model:');
    for m = 1:numel(modelNames)
        fprintf('  %s: %.4f\n', modelNames{m}, xp(m));
    end

    disp('Protected Exceedance Probability (pxp) for each model:');
     for m = 1:numel(modelNames)
        fprintf('  %s: %.4f\n', modelNames{m}, pxp(m));
    end

    fprintf('Bayes Omnibus Risk (BOR - prob. that models are equally likely): %.4f\n', bor);

     % --- Save results to CSV for standard models ---
    % Ensure modelNames is a column vector for table creation
    if isrow(modelNames)
        modelNamesCol = modelNames';
    else
        modelNamesCol = modelNames;
    end
    % Ensure EP and PEP are column vectors
    exp_rCol = exp_r';
    xpCol = xp';
    pxpCol = pxp';
    emfCol = out.Ef;
    emfVarCol = out.Vf;

    resultsTable = table(modelNamesCol, exp_rCol, xpCol, pxpCol, emfCol, emfVarCol, ...
        'VariableNames', {'ModelName', ...
        'Expected Posterior Probability', ...
        'ExceedanceProbability', ...
        'ProtectedExceedanceProbability', ...
        'EstimatedModelFrequency', ...
        'EstimatedModelFrequencyVariance'});
    writetable(resultsTable, outputCsvFile_standard_AIC_approx);
    fprintf('Standard model BMS results with AIC approx saved to %s\n', ...
        outputCsvFile_standard_AIC_approx);

catch ME_bms_aic_approx
    fprintf('Error during BMS execution for standard models with AIC approx:\n');
    rethrow(ME_bms_aic_approx);
end

% --- Run Bayesian DDM Model Selection with BIC approx ---

fprintf('\n--- Running Bayesian DDM Model Selection wit BIC approx ---\n');

try
    [alpha, exp_r, xp, pxp, bor] = spm_BMS(BIC_approx_LME_ddm_matrix);

    [posterior, out] = VBA_groupBMC(BIC_approx_LME_ddm_matrix');

    disp('Exceedance Probabilities according to VBA toolbox:');
    disp(out.ep');

    %disp('Alpha (Dirichlet parameters for model posterior):');
    %disp(alpha'); % Transpose for easier reading if many models

    disp('Estimated Model Frequency:');
    disp(out.Ef);

    disp('Estimated Model Frequency Variance:');
    disp(out.Vf);

    disp('Expected Posterior Probability (exp_r) for each model:');
    for m = 1:numel(ddmModelNames)
        fprintf('  %s: %.4f\n', ddmModelNames{m}, exp_r(m));
    end

    disp('Exceedance Probability (xp) for each model:');
    for m = 1:numel(ddmModelNames)
        fprintf('  %s: %.4f\n', ddmModelNames{m}, xp(m));
    end

    disp('Protected Exceedance Probability (pxp) for each model:');
     for m = 1:numel(ddmModelNames)
        fprintf('  %s: %.4f\n', ddmModelNames{m}, pxp(m));
    end

    fprintf('Bayes Omnibus Risk (BOR - prob. that models are equally likely): %.4f\n', bor);

     % --- Save results to CSV for standard models ---
    % Ensure modelNames is a column vector for table creation
    if isrow(ddmModelNames)
        modelNamesCol = ddmModelNames';
    else
        modelNamesCol = ddmModelNames;
    end
    % Ensure EP and PEP are column vectors
    exp_rCol = exp_r';
    xpCol = xp';
    pxpCol = pxp';
    emfCol = out.Ef;
    emfVarCol = out.Vf;

    resultsTable = table(modelNamesCol, exp_rCol, xpCol, pxpCol, emfCol, emfVarCol, ...
        'VariableNames', {'ModelName', ...
        'Expected Posterior Probability', ...
        'ExceedanceProbability', ...
        'ProtectedExceedanceProbability', ...
        'EstimatedModelFrequency', ...
        'EstimatedModelFrequencyVariance'});
    writetable(resultsTable, outputCsvFile_ddm_BIC_approx);
    fprintf('DDM models BMS results with BIC approx saved to %s\n', ...
        outputCsvFile_ddm_BIC_approx);

catch ME_bms_bic_approx_ddm
    fprintf('Error during BMS execution for ddm models with BIC approx:\n');
    rethrow(ME_bms_bic_approx_ddm);
end

% --- Run Bayesian DDM Model Selection with AIC approx ---

fprintf('\n--- Running Bayesian DDM Model Selection wit AIC approx ---\n');

try
    [alpha, exp_r, xp, pxp, bor] = spm_BMS(AIC_approx_LME_ddm_matrix);

    [posterior, out] = VBA_groupBMC(AIC_approx_LME_ddm_matrix');

    disp('Exceedance Probabilities according to VBA toolbox:');
    disp(out.ep');

    %disp('Alpha (Dirichlet parameters for model posterior):');
    %disp(alpha'); % Transpose for easier reading if many models

    disp('Estimated Model Frequency:');
    disp(out.Ef);

    disp('Estimated Model Frequency Variance:');
    disp(out.Vf);

    disp('Expected Posterior Probability (exp_r) for each model:');
    for m = 1:numel(ddmModelNames)
        fprintf('  %s: %.4f\n', ddmModelNames{m}, exp_r(m));
    end

    disp('Exceedance Probability (xp) for each model:');
    for m = 1:numel(ddmModelNames)
        fprintf('  %s: %.4f\n', ddmModelNames{m}, xp(m));
    end

    disp('Protected Exceedance Probability (pxp) for each model:');
     for m = 1:numel(ddmModelNames)
        fprintf('  %s: %.4f\n', ddmModelNames{m}, pxp(m));
    end

    fprintf('Bayes Omnibus Risk (BOR - prob. that models are equally likely): %.4f\n', bor);

     % --- Save results to CSV for standard models ---
    % Ensure ddmModelNames is a column vector for table creation
    if isrow(ddmModelNames)
        modelNamesCol = ddmModelNames';
    else
        modelNamesCol = ddmModelNames;
    end
    % Ensure EP and PEP are column vectors
    exp_rCol = exp_r';
    xpCol = xp';
    pxpCol = pxp';
    emfCol = out.Ef;
    emfVarCol = out.Vf;

    %disp(emfVarCol)

    resultsTable = table(modelNamesCol, exp_rCol, xpCol, pxpCol, emfCol, emfVarCol, ...
        'VariableNames', {'ModelName', ...
        'Expected Posterior Probability', ...
        'ExceedanceProbability', ...
        'ProtectedExceedanceProbability', ...
        'EstimatedModelFrequency', ...
        'EstimatedModelFrequencyVariance'});
    writetable(resultsTable, outputCsvFile_ddm_AIC_approx);
    fprintf('DDM models BMS results with BIC approx saved to %s\n', ...
        outputCsvFile_ddm_AIC_approx);

catch ME_bms_aic_approx_ddm
    fprintf('Error during BMS execution for ddm models with AIC approx:\n');
    rethrow(ME_bms_aic_approx_ddm);
end


% --- Run Bayesian DDM Family Model Selection with BIC approx ---

fprintf('\n--- Running Bayesian DDM Family Model Selection wit BIC approx ---\n');

try

    % Test RL_ddm vs AI_ddm
    options.families = {[1], [2,3,4,5]};

    family.names = {'HRL','AIF','AIF','AIF','AIF'};
    family.partition = [1,2,2,2,2];
    family.infer = 'RFX';


    [family, model] = spm_compare_families(BIC_approx_LME_ddm_matrix,family);

    [posterior, out] = VBA_groupBMC(BIC_approx_LME_ddm_matrix',options);

    disp('Exceedance Probabilities according to VBA toolbox:');
    disp(out.ep');

    %disp('Alpha (Dirichlet parameters for model posterior):');
    %disp(alpha'); % Transpose for easier reading if many models

    disp('Estimated Model Frequency:');
    disp(out.Ef);

    disp('Estimated Model Family Frequency:');
    disp(out.families.Ef);

    disp('Estimated Model Frequency Variance:');
    disp(out.Vf);

    disp('Expected Posterior Probability (exp_r) for each model:');
    for m = 1:numel(ddmModelNames)
        fprintf('  %s: %.4f\n', ddmModelNames{m}, model.exp_r(m));
    end

    disp('Expected Posterior Probability (exp_r) for each family:');
    disp(family.exp_r);

    disp('Exceedance Probability (xp) for each family:');
    disp(family.xp);

    disp('Estimated Model Family Frequency:');
    disp(out.families.Ef);

    disp('Estimated Model Family Frequency Variance:');
    disp(out.families.Vf);

     % --- Save results to CSV for ddm model families ---
    % Ensure ddmModelNames is a column vector for table creation
    if isrow(ddmModelNames)
        modelNamesCol = ddmModelNames';
    else
        modelNamesCol = ddmModelNames;
    end
    % Ensure EP and PEP are column vectors
    exp_rCol = model.exp_r';

    resultsTable = table(modelNamesCol, exp_rCol, ...
        'VariableNames', {'ModelName', ...
        'Expected Posterior Probability'});
    writetable(resultsTable, outputCsvFile_ddm_families_BIC_approx);
    fprintf('DDM family models BMS results with BIC approx saved to %s\n', ...
        outputCsvFile_ddm_families_BIC_approx);

    % --- Save results to CSV for ddm model families comparison---

    %fam_exp_rCol = family.exp_r';
    %fam_xpCol = family.xp';
    %fam_efCol = out.families.Ef;
    %fam_efVarCol = out.families.Vf;

    %resultsTableComp = table(modelNamesCol, fam_exp_rCol, fam_xpCol, fam_efCol, fam_efVarCol,...
    %    'VariableNames', {'ModelName', ...
    %    'FamilyExpectedPosteriorProbability', ...
    %    'FamilyExceedanceProbability', ...
    %    'FamilyEstimatedFrequency', ...
    %    'FamilyEstimatedFrequencyVariance'});
    %writetable(resultsTableComp, outputCsvFile_ddm_families_comparison_BIC_approx);
    %fprintf('DDM model family comparison BMS results with BIC approx saved to %s\n', ...
    %    outputCsvFile_ddm_families_comparison_BIC_approx);


catch ME_bms_bic_approx_ddm_families
    fprintf('Error during BMS execution for ddm models with BIC approx:\n');
    rethrow(ME_bms_bic_approx_ddm_families);
end

% --- Run Bayesian DDM Family Model Selection with AIC approx ---

fprintf('\n--- Running Bayesian DDM Family Model Selection wit AIC approx ---\n');

try

    % Test RL_ddm vs AI_ddm
    options.families = {[1], [2,3,4,5]};

    family.names = {'HRL','AIF','AIF','AIF','AIF'};
    family.infer = 'RFX';
    family.partition = [1,2,2,2,2];

    [family, model] = spm_compare_families(AIC_approx_LME_ddm_matrix,family);

    [posterior, out] = VBA_groupBMC(AIC_approx_LME_ddm_matrix',options);

    disp('Exceedance Probabilities according to VBA toolbox:');
    disp(out.ep');

    %disp('Alpha (Dirichlet parameters for model posterior):');
    %disp(alpha'); % Transpose for easier reading if many models

    disp('Estimated Model Frequency:');
    disp(out.Ef);

    disp('Estimated Model Frequency Variance:');
    disp(out.Vf);

    disp('Expected Posterior Probability (exp_r) for each model:');
    for m = 1:numel(ddmModelNames)
        fprintf('  %s: %.4f\n', ddmModelNames{m}, model.exp_r(m));
    end

    disp('Expected Posterior Probability (exp_r) for each family:');
    disp(family.exp_r);

    disp('Exceedance Probability (xp) for each family:');
    disp(family.xp);

    disp('Estimated Model Family Frequency:');
    disp(out.families.Ef);

    disp('Estimated Model Family Frequency Variance:');
    disp(out.families.Vf);

     % --- Save results to CSV for ddm model families ---
    % Ensure ddmModelNames is a column vector for table creation
    if isrow(ddmModelNames)
        modelNamesCol = ddmModelNames';
    else
        modelNamesCol = ddmModelNames;
    end
    % Ensure EP and PEP are column vectors
    exp_rCol = model.exp_r';

    resultsTable = table(modelNamesCol, exp_rCol, ...
        'VariableNames', {'ModelName', ...
        'Expected Posterior Probability'});
    writetable(resultsTable, outputCsvFile_ddm_families_AIC_approx);
    fprintf('DDM family models BMS results with AIC approx saved to %s\n', ...
        outputCsvFile_ddm_families_AIC_approx);

    % --- Save results to CSV for ddm model families comparison---

    %fam_exp_rCol = family.exp_r';
    %fam_xpCol = family.xp';
    %fam_efCol = out.families.Ef;
    %fam_efVarCol = out.families.Vf;

    %resultsTable = table(modelNamesCol, fam_exp_rCol, fam_xpCol, fam_efCol, fam_efVarCol,...
    %    'VariableNames', {'ModelName', ...
    %    'FamilyExpectedPosteriorProbability', ...
    %    'FamilyExceedanceProbability', ...
    %    'FamilyEstimatedFrequency', ...
    %    'FamilyEstimatedFrequencyVariance'});
    %writetable(resultsTable, outputCsvFile_ddm_families_comparison_AIC_approx);
    %fprintf('DDM model family comparison BMS results with AIC approx saved to %s\n', ...
    %    outputCsvFile_ddm_families_comparison_AIC_approx);

catch ME_bms_aic_approx_ddm_families
    fprintf('Error during BMS execution for ddm models with AIC approx:\n');
    rethrow(ME_bms_aic_approx_ddm_families);
end