function [net,report] = ...
    adni_resnet3d18_set_training_scope(net,scope,options)
%ADNI_RESNET3D18_SET_TRAINING_SCOPE Freeze or unfreeze 3-D ResNet weights.

arguments
    net (1,1) dlnetwork
    scope (1,1) string {mustBeMember(scope, ...
        ["head-only","stage4-head","all"])}
    options.HeadFactor (1,1) double {mustBeNonnegative} = 1
    options.BackboneFactor (1,1) double {mustBeNonnegative} = 1
end

learnables = net.Learnables;
factors = zeros(height(learnables),1);
for row = 1:height(learnables)
    layerName = learnables.Layer(row);
    parameterName = learnables.Parameter(row);
    if layerName == "fc"
        factor = options.HeadFactor;
    elseif scope == "head-only"
        factor = 0;
    elseif scope == "stage4-head"
        if startsWith(layerName,"stack4:")
            factor = options.BackboneFactor;
        else
            factor = 0;
        end
    else
        factor = options.BackboneFactor;
    end
    net = setLearnRateFactor( ...
        net,layerName,parameterName,factor);
    factors(row) = factor;
end

assert(any(learnables.Layer == "fc"),"NMA:MissingClassifier", ...
    "The 3-D ResNet classifier layer 'fc' is missing.");
if scope == "head-only"
    assert(all(factors(learnables.Layer ~= "fc") == 0) && ...
        all(factors(learnables.Layer == "fc") == ...
        options.HeadFactor), ...
        "NMA:FreezeVerificationFailed", ...
        "Head-only training scope was not applied exactly.");
elseif scope == "stage4-head"
    stage4Mask = startsWith(learnables.Layer,"stack4:");
    assert(any(stage4Mask) && ...
        all(factors(stage4Mask) == options.BackboneFactor) && ...
        all(factors(learnables.Layer == "fc") == ...
        options.HeadFactor) && ...
        all(factors(~stage4Mask & ...
        learnables.Layer ~= "fc") == 0), ...
        "NMA:Stage4FreezeVerificationFailed", ...
        "Stage-4-plus-head training scope was not applied exactly.");
else
    assert(all(factors(learnables.Layer ~= "fc") == ...
        options.BackboneFactor) && ...
        all(factors(learnables.Layer == "fc") == ...
        options.HeadFactor), ...
        "NMA:UnfreezeVerificationFailed", ...
        "Full training scope was not applied exactly.");
end

report = struct;
report.Scope = scope;
report.HeadFactor = options.HeadFactor;
report.BackboneFactor = options.BackboneFactor;
report.NumLearnableTensors = height(learnables);
report.NumFrozenTensors = sum(factors == 0);
report.NumTrainableTensors = sum(factors > 0);
report.Layer = learnables.Layer;
report.Parameter = learnables.Parameter;
report.Factor = factors;
report.CreatedAt = datetime("now",TimeZone="local");
end
