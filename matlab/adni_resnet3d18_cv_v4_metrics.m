function [metrics,confusion,predictedGroup,normalizedScores] = ...
    adni_resnet3d18_cv_v4_metrics(trueGroup,scores,options)
%ADNI_RESNET3D18_CV_V4_METRICS Frozen participant-level v4 endpoints.

arguments
    trueGroup (:,1) string
    scores (:,3) double
    options.ClassNames (:,1) string = ["AD";"CN";"MCI"]
    options.NumCalibrationBins (1,1) double ...
        {mustBeInteger,mustBePositive} = 10
end

classNames = options.ClassNames;
assert(numel(classNames) == 3 && ...
    numel(unique(classNames)) == 3 && ...
    height(scores) == numel(trueGroup) && ~isempty(trueGroup) && ...
    all(ismember(trueGroup,classNames)) && ...
    all(isfinite(scores),"all") && all(scores >= 0,"all"), ...
    "NMA:InvalidCVV4MetricInput", ...
    "Truth and scores do not satisfy the frozen v4 metric contract.");
rowSums = sum(scores,2);
assert(all(rowSums > 0),"NMA:ZeroCVV4ScoreSum", ...
    "Every prediction row must have positive total score.");
normalizedScores = scores./rowSums;
[~,predictionIndex] = max(normalizedScores,[],2);
predictedGroup = classNames(predictionIndex);

numClasses = numel(classNames);
confusion = zeros(numClasses,numClasses);
truthIndex = zeros(numel(trueGroup),1);
for row = 1:numel(trueGroup)
    trueIndex = find(classNames == trueGroup(row));
    assert(isscalar(trueIndex),"NMA:UnknownCVV4MetricClass", ...
        "A prediction has an unknown true class.");
    truthIndex(row) = trueIndex;
    confusion(trueIndex,predictionIndex(row)) = ...
        confusion(trueIndex,predictionIndex(row))+1;
end

truePositive = diag(confusion);
support = sum(confusion,2);
predictedCount = sum(confusion,1).';
recall = truePositive./max(support,1);
precision = truePositive./max(predictedCount,1);
f1 = 2*precision.*recall./max(precision+recall,eps);
accuracy = sum(truePositive)/sum(confusion,"all");
balancedAccuracy = mean(recall);
macroF1 = mean(f1);

oneHot = zeros(numel(truthIndex),numClasses);
oneHot(sub2ind(size(oneHot),(1:numel(truthIndex)).',truthIndex)) = 1;
trueProbabilities = normalizedScores(oneHot == 1);
negativeLogLikelihood = -mean(log(max(trueProbabilities,eps)));
brierScore = mean(sum((normalizedScores-oneHot).^2,2));
perClassAUROC = zeros(numClasses,1);
for classIndex = 1:numClasses
    perClassAUROC(classIndex) = binary_auroc( ...
        truthIndex == classIndex,normalizedScores(:,classIndex));
end
macroAUROC = mean(perClassAUROC,"omitnan");

[confidence,calibrationPrediction] = max(normalizedScores,[],2);
correct = calibrationPrediction == truthIndex;
edges = linspace(0,1,options.NumCalibrationBins+1);
ece = 0;
for bin = 1:options.NumCalibrationBins
    if bin < options.NumCalibrationBins
        members = confidence >= edges(bin) & ...
            confidence < edges(bin+1);
    else
        members = confidence >= edges(bin) & ...
            confidence <= edges(bin+1);
    end
    if any(members)
        ece = ece+mean(members)*abs( ...
            mean(correct(members))-mean(confidence(members)));
    end
end

metrics = struct;
metrics.Accuracy = accuracy;
metrics.BalancedAccuracy = balancedAccuracy;
metrics.MacroF1 = macroF1;
metrics.RecallAD = recall(1);
metrics.RecallCN = recall(2);
metrics.RecallMCI = recall(3);
metrics.MacroAUROC = macroAUROC;
metrics.NegativeLogLikelihood = negativeLogLikelihood;
metrics.BrierScore = brierScore;
metrics.ECE = ece;
metrics.PerClassAUROC = perClassAUROC;
end

function value = binary_auroc(truth,score)
truth = logical(truth(:));
score = double(score(:));
positive = sum(truth);
negative = sum(~truth);
if positive == 0 || negative == 0
    value = NaN;
    return
end
ranks = tied_ranks(score);
value = (sum(ranks(truth))-positive*(positive+1)/2)/ ...
    (positive*negative);
end

function ranks = tied_ranks(values)
[sorted,order] = sort(values);
ranksSorted = zeros(size(sorted));
first = 1;
while first <= numel(sorted)
    last = first;
    while last < numel(sorted) && sorted(last+1) == sorted(first)
        last = last+1;
    end
    ranksSorted(first:last) = 0.5*(first+last);
    first = last+1;
end
ranks = zeros(size(values));
ranks(order) = ranksSorted;
end
