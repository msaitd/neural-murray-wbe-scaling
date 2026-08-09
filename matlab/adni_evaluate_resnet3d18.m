function evaluation = ...
    adni_evaluate_resnet3d18(net,datasetManifest,options)
%ADNI_EVALUATE_RESNET3D18 Evaluate one 3-D volume per ADNI participant.

arguments
    net (1,1) dlnetwork
    datasetManifest (1,1) string
    options.Split (1,1) string = "test"
    options.ExecutionEnvironment (1,1) string = "auto"
    options.BootstrapReplicates (1,1) double ...
        {mustBeInteger,mustBeNonnegative} = 1000
    options.RandomSeed (1,1) double {mustBeInteger} = 20260727
    options.OutputPrefix (1,1) string = ""
end

manifest = readtable( ...
    datasetManifest,Delimiter=",",TextType="string", ...
    VariableNamingRule="preserve");
manifest = manifest(manifest.split == options.Split,:);
assert(height(manifest) > 0,"NMA:EmptyEvaluationSplit", ...
    "No 3-D rows found for split %s.",options.Split);
assert(all(isfile(manifest.matPath)),"NMA:MissingPrepared3DVolume", ...
    "At least one prepared 3-D volume is missing.");
assert(numel(unique(manifest.subject)) == height(manifest), ...
    "NMA:Duplicate3DSubject", ...
    "The 3-D manifest must contain one row per participant per split.");

useGPU = options.ExecutionEnvironment == "gpu" || ...
    (options.ExecutionEnvironment == "auto" && canUseGPU);
classNames = ["AD";"CN";"MCI"];
numSubjects = height(manifest);
scores = zeros(numSubjects,numel(classNames));
for row = 1:numSubjects
    volume = single(adni_read_3d_mat(manifest.matPath(row)));
    if useGPU
        volume = gpuArray(volume);
    end
    prediction = predict(net,dlarray(volume,"SSSCB"));
    values = extractdata(prediction);
    if isa(values,"gpuArray")
        values = gather(values);
    end
    scores(row,:) = double(values(:)).';
end

trueGroup = manifest.group;
[~,predictionIndex] = max(scores,[],2);
predictedGroup = classNames(predictionIndex);
confusion = confusion_from_labels(trueGroup,predictedGroup,classNames);
[accuracy,balancedAccuracy,macroF1,perClass] = ...
    classification_metrics(confusion,classNames);
[accuracyInterval,balancedInterval,macroF1Interval] = ...
    bootstrap_intervals( ...
    trueGroup,predictedGroup,classNames, ...
    options.BootstrapReplicates,options.RandomSeed);

subjectTable = table( ...
    manifest.subject,trueGroup,predictedGroup, ...
    scores(:,1),scores(:,2),scores(:,3), ...
    VariableNames=[ ...
        "Subject","TrueGroup","PredictedGroup", ...
        "ScoreAD","ScoreCN","ScoreMCI"]);
evaluation = struct;
evaluation.Split = options.Split;
evaluation.ClassNames = classNames;
evaluation.NumSubjects = numSubjects;
evaluation.SubjectPredictions = subjectTable;
evaluation.ConfusionMatrix = confusion;
evaluation.Accuracy = accuracy;
evaluation.BalancedAccuracy = balancedAccuracy;
evaluation.MacroF1 = macroF1;
evaluation.PerClass = perClass;
evaluation.Accuracy95CI = accuracyInterval;
evaluation.BalancedAccuracy95CI = balancedInterval;
evaluation.MacroF195CI = macroF1Interval;
evaluation.BootstrapReplicates = options.BootstrapReplicates;
evaluation.RandomSeed = options.RandomSeed;
evaluation.CreatedAt = datetime("now",TimeZone="local");

if strlength(options.OutputPrefix) > 0
    writetable(subjectTable,options.OutputPrefix+"_subjects.csv", ...
        Encoding="UTF-8");
    save(options.OutputPrefix+"_evaluation.mat","evaluation","-v7.3");
end
end

function confusion = confusion_from_labels(truth,prediction,classNames)
confusion = zeros(numel(classNames),numel(classNames));
for row = 1:numel(truth)
    trueIndex = find(classNames == truth(row));
    predictedIndex = find(classNames == prediction(row));
    assert(isscalar(trueIndex) && isscalar(predictedIndex), ...
        "NMA:UnknownEvaluationClass","Unknown class label.");
    confusion(trueIndex,predictedIndex) = ...
        confusion(trueIndex,predictedIndex)+1;
end
end

function [accuracy,balancedAccuracy,macroF1,perClass] = ...
    classification_metrics(confusion,classNames)
truePositive = diag(confusion);
support = sum(confusion,2);
predicted = sum(confusion,1).';
recall = truePositive./max(support,1);
precision = truePositive./max(predicted,1);
f1 = 2*precision.*recall./max(precision+recall,eps);
accuracy = sum(truePositive)/sum(confusion,"all");
balancedAccuracy = mean(recall);
macroF1 = mean(f1);
perClass = table( ...
    classNames,support,precision,recall,f1, ...
    VariableNames=["Class","Support","Precision","Recall","F1"]);
end

function [accuracyInterval,balancedInterval,macroF1Interval] = ...
    bootstrap_intervals(truth,prediction,classNames,replicates,seed)
if replicates == 0
    accuracyInterval = [NaN NaN];
    balancedInterval = [NaN NaN];
    macroF1Interval = [NaN NaN];
    return
end
rng(seed,"twister");
numSubjects = numel(truth);
metrics = zeros(replicates,3);
for replicate = 1:replicates
    indices = randi(numSubjects,numSubjects,1);
    confusion = confusion_from_labels( ...
        truth(indices),prediction(indices),classNames);
    [metrics(replicate,1),metrics(replicate,2), ...
        metrics(replicate,3)] = ...
        classification_metrics(confusion,classNames);
end
metrics = sort(metrics,1);
lowerIndex = max(1,round(0.025*replicates));
upperIndex = min(replicates,round(0.975*replicates));
accuracyInterval = metrics([lowerIndex upperIndex],1).';
balancedInterval = metrics([lowerIndex upperIndex],2).';
macroF1Interval = metrics([lowerIndex upperIndex],3).';
end
