function [net,report] = ...
    adni_resnet3d18_reset_classifier(net,randomSeed)
%ADNI_RESNET3D18_RESET_CLASSIFIER Apply a reproducible He-normal head.

arguments
    net (1,1) dlnetwork
    randomSeed (1,1) double {mustBeInteger}
end

learnablesBefore = net.Learnables;
classifier = getLayer(net,"fc");
previousState = rng;
cleanup = onCleanup(@() rng(previousState));
rng(randomSeed,"twister");
classifier.Weights = sqrt(single(2/classifier.InputSize))* ...
    randn(classifier.OutputSize,classifier.InputSize,"single");
classifier.Bias = zeros(size(classifier.Bias),"single");
net = replaceLayer(net,"fc",classifier);
net = initialize(net);
clear cleanup
rng(previousState);

learnablesAfter = net.Learnables;
assert(height(learnablesBefore) == height(learnablesAfter), ...
    "NMA:ClassifierResetChangedLearnables", ...
    "Classifier reset changed the learnable table height.");
backboneExact = true;
for row = 1:height(learnablesBefore)
    if learnablesBefore.Layer(row) == "fc"
        continue
    end
    matching = find( ...
        learnablesAfter.Layer == learnablesBefore.Layer(row) & ...
        learnablesAfter.Parameter == learnablesBefore.Parameter(row));
    assert(isscalar(matching), ...
        "NMA:ClassifierResetChangedLearnables", ...
        "Could not uniquely match %s/%s after classifier reset.", ...
        learnablesBefore.Layer(row), ...
        learnablesBefore.Parameter(row));
    before = extractdata(learnablesBefore.Value{row});
    after = extractdata(learnablesAfter.Value{matching});
    backboneExact = backboneExact && isequal(before,after);
end
assert(backboneExact,"NMA:ClassifierResetChangedBackbone", ...
    "Classifier reset unexpectedly changed backbone parameters.");

report = struct;
report.RandomSeed = randomSeed;
report.Initializer = "He-normal-sqrt(2/fan-in)";
report.OutputSize = classifier.OutputSize;
report.InputSize = classifier.InputSize;
report.BackboneExact = backboneExact;
report.WeightMean = mean(double(classifier.Weights),"all");
report.WeightStandardDeviation = ...
    std(double(classifier.Weights),0,"all");
report.CreatedAt = datetime("now",TimeZone="local");
end
