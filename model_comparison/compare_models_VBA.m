% ----------- Import Log Model Evidence (LME) data-----------%
lme_FilePath = 'LMEs.csv'; 
lme_ddm_FilePath = 'LMEs_ddm.csv'; 
%participantIdColumnName = 'ParticipantID';
modelNames = {'RL', 'AI0','AI1', 'AI2', 'AI3'};
%ddmModelNames = {}

lmeTable = readtable(lme_FilePath);
lme_ddmTable = readtable(lme_ddm_FilePath);

allColNames_lme = lmeTable.Properties.VariableNames;
%modelNames = allColNames_lme(~'ParticipantID');

disp(modelNames'); % Transpose for better display

allColNames_lme_ddm = lme_ddmTable.Properties.VariableNames;
%ddmModelNames = allColNames_lme_ddm(~'ParticipantID');
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
lmeSubTable = lmeTable(:, modelNames);
LME_matrix = table2array(lmeSubTable)';

lme_ddmSubTable = lme_ddmTable(:, ddmModelNames);
LME_ddm_matrix = table2array(lme_ddmSubTable)';

% --- Define output CSV filenames ---
outputCsvFile_standard = 'bms_results_standard_models.csv';
outputCsvFile_ddm = 'bms_results_ddm_models.csv';
outputCsvFile_ddm_families = 'bms_results_ddm_model_families.csv';

% --- Run SPM Bayesian Model Selection (BMS) ---

%fprintf('\n--- Running VBA Bayesian Model Selection ---\n');


try
    [posterior, out] = VBA_groupBMC(LME_matrix);
    disp(posterior.r)

    fprintf('\n--- BMC Results ---\n');
    disp('Estimated model frequencies:');
    disp(out.Ef');
    disp('Error Estimated model Frequencies:');
    disp(out.Vf');
    

    %disp('Expected Posterior Probability (exp_r) for each model:');
    %for m = 1:numel(modelNames)
    %    fprintf('  %s: %.4f\n', modelNames{m}, exp_r(m));
    %end

    disp('Exceedance Probabilities:');
    disp(out.ep');
    
    PEP = (1-out.bor)*out.ep + out.bor/length(out.ep);

    disp('Protected Exceedance Probabilities:');
    disp(PEP');

     % --- Save results to CSV for standard models ---
    % Ensure modelNames is a column vector for table creation
    if isrow(modelNames)
        modelNamesCol = modelNames';
    else
        modelNamesCol = modelNames;
    end
    % Ensure EP and PEP are column vectors
    epCol = out.ep';
    pepCol = PEP';
    emfCol = out.Ef;
    emfVarCol = out.Vf;

    resultsTable = table(modelNamesCol, epCol, pepCol, emfCol, emfVarCol, ...
        'VariableNames', {'ModelName', 'ExceedanceProbability', 'ProtectedExceedanceProbability', 'EstimatedModelFrequency','EstimatedModelFrequencyVariance'});
    writetable(resultsTable, outputCsvFile_standard);
    fprintf('Standard model BMS results saved to %s\n', outputCsvFile_standard);



catch ME_bms
    fprintf('Error during spm_BMS execution:\n');
    rethrow(ME_bms);
end


fprintf('\n--- Running VBA Bayesian Model Selection for DDM models ---\n');


try
    [posterior_ddm, out_ddm] = VBA_groupBMC(LME_ddm_matrix);


    fprintf('\n--- BMC Results ---\n');
    disp('Estimated model frequencies:');
    disp(out_ddm.Ef');
    disp('Error Estimated model Frequencies:');
    disp(out.Vf');

    %disp('Expected Posterior Probability (exp_r) for each model:');
    %for m = 1:numel(modelNames)
    %    fprintf('  %s: %.4f\n', modelNames{m}, exp_r(m));
    %end

    disp('Exceedance Probabilities:');
    disp(out_ddm.ep');
    
    PEP_ddm = (1-out_ddm.bor)*out_ddm.ep + out_ddm.bor/length(out_ddm.ep);

    disp('Protected Exceedance Probabilities:');
    disp(PEP_ddm');

    % --- Save results to CSV for DDM models ---
    % Ensure ddmModelNames is a column vector for table creation
    if isrow(ddmModelNames)
        ddmModelNamesCol = ddmModelNames';
    else
        ddmModelNamesCol = ddmModelNames;
    end
    % Ensure EP and PEP are column vectors
    ep_ddmCol = out_ddm.ep';
    pep_ddmCol = PEP_ddm';
    emf_ddmCol = out_ddm.Ef;
    emfVar_ddmCol = out_ddm.Vf;

    resultsTable_ddm = table(ddmModelNamesCol, ep_ddmCol, pep_ddmCol, emf_ddmCol,emfVar_ddmCol, ...
        'VariableNames', {'ModelName', 'ExceedanceProbability', 'ProtectedExceedanceProbability', 'EstimatedModelFrequency','EstimatedModelFrequencyVariance'});
    writetable(resultsTable_ddm, outputCsvFile_ddm);
    fprintf('DDM model BMS results saved to %s\n', outputCsvFile_ddm);


catch ME_bms_ddm
    fprintf('Error during spm_BMS execution:\n');
    rethrow(ME_bms_ddm);
end

fprintf('\n--- Running VBA Bayesian Model Selection for DDM models with group families ---\n');


try
    % Test RL_ddm vs AI_ddm
    %options.families = {[1], [2,3,4,5],[6],[7,8,9,10]};

    % Test RL_ddm vs AI_ddm
    %options.families = {[1,6], [2,3,4,5,7,8,9,10]};
    options.families = {[1], [2,3,4,5]};

    [posterior_ddm, out_ddm] = VBA_groupBMC(LME_ddm_matrix, options);


    fprintf('\n--- BMC Results ---\n');
    disp('Estimated model frequencies:');
    disp(out_ddm.Ef');

    %disp('Expected Posterior Probability (exp_r) for each model:');
    %for m = 1:numel(modelNames)
    %    fprintf('  %s: %.4f\n', modelNames{m}, exp_r(m));
    %end

    disp('Exceedance Probabilities:');
    disp(out_ddm.ep');

    disp('Model Family Exceedance Probabilities:');
    disp(out_ddm.families.ep');

  
    % --- Save results to CSV for DDM model families analysis ---
    % Ensure ddmModelNames is a column vector for table creation
    %if isrow(ddmModelNames)
    %    ddmModelNamesCol = ddmModelNames';
    %else
    %    ddmModelNamesCol = ddmModelNames;
    %end
    % Ensure EP and PEP are column vectors
    %ep_ddmCol = out_ddm.ep';
    %pep_ddmCol = PEP_ddm';

    %resultsTable_ddm = table(ddmModelNamesCol, ep_ddmCol, pep_ddmCol, ...
    %    'VariableNames', {'ModelName', 'ExceedanceProbability', 'ProtectedExceedanceProbability'});
    %writetable(resultsTable_ddm, outputCsvFile_ddm);
    %fprintf('DDM model BMS results saved to %s\n', outputCsvFile_ddm);


catch ME_bms_ddm_families
    fprintf('Error during spm_BMS execution:\n');
    rethrow(ME_bms_ddm_families);
end


% --- End of Script ---