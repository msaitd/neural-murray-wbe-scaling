function [compressedNet,report] = ...
    nma_resnet3d18_rebuild(net,allocation,calibration,options)
%NMA_RESNET3D18_REBUILD Apply NMA widths with inherited 3-D weights.

arguments
    net (1,1) dlnetwork
    allocation (1,1) struct
    calibration (1,1) struct
    options.ScoreEstimator (1,1) string = "allocation"
end

spec = nma_resnet3d18_spec(net);
assert(isfield(allocation,"TargetWidths"),"NMA:InvalidAllocation", ...
    "The allocation input must contain TargetWidths.");
targetWidths = double(allocation.TargetWidths(:));
assert(numel(targetWidths) == 4 && ...
    all(isfinite(targetWidths) & targetWidths == round(targetWidths) & ...
    targetWidths > 0),"NMA:InvalidTargetWidths", ...
    "TargetWidths must contain four positive integers.");
assert(all(targetWidths <= spec.BaselineWidths), ...
    "NMA:WideningNotSupported", ...
    "Inherited-weight surgery cannot exceed baseline widths.");

estimator = lower(options.ScoreEstimator);
if estimator == "allocation"
    assert(isfield(allocation,"Estimator"),"NMA:MissingEstimator", ...
        "Allocation does not record the demand estimator.");
    estimator = lower(string(allocation.Estimator));
end

stageMasks = cell(4,1);
for stage = 1:4
    score = probe_score( ...
        calibration,spec.DemandProbeNames(stage),estimator, ...
        spec.BaselineWidths(stage));
    stageMasks{stage} = select_channels(score,targetWidths(stage));
end

hiddenMasks = struct;
for probe = reshape(spec.HiddenProbeNames,1,[])
    stage = stage_index_from_name(probe);
    score = probe_score( ...
        calibration,probe,estimator,spec.BaselineWidths(stage));
    hiddenMasks.(matlab.lang.makeValidName(probe)) = ...
        select_channels(score,targetWidths(stage));
end

internalNormalization = "batch";
for layerIndex = 1:numel(net.Layers)
    if isa(net.Layers(layerIndex), ...
            "nnet.cnn.layer.GroupNormalizationLayer")
        internalNormalization = "group";
        break
    end
end
compressedNet = adni_resnet3d18_create( ...
    targetWidths.',InputSize=spec.InputSize, ...
    NumClasses=getLayer(net,"fc").OutputSize, ...
    InternalNormalization=internalNormalization);
compressedNet = replaceLayer( ...
    compressedNet,spec.InputName,getLayer(net,spec.InputName));
replacedConvolutions = strings(0,1);
replacedNormalizations = strings(0,1);

for layerIndex = 1:numel(net.Layers)
    oldLayer = net.Layers(layerIndex);
    layerName = string(oldLayer.Name);

    if isa(oldLayer,"nnet.cnn.layer.Convolution3DLayer")
        [inputMask,outputMask] = convolution_masks( ...
            layerName,stageMasks,hiddenMasks);
        newLayer = sliced_convolution(oldLayer,inputMask,outputMask);
        compressedNet = replaceLayer(compressedNet,layerName,newLayer);
        replacedConvolutions(end+1,1) = layerName; %#ok<AGROW>

    elseif isa(oldLayer,"nnet.cnn.layer.BatchNormalizationLayer")
        mask = normalization_mask(layerName,stageMasks,hiddenMasks);
        newLayer = sliced_batch_normalization(oldLayer,mask);
        compressedNet = replaceLayer(compressedNet,layerName,newLayer);
        replacedNormalizations(end+1,1) = layerName; %#ok<AGROW>

    elseif isa(oldLayer,"nnet.cnn.layer.GroupNormalizationLayer")
        mask = normalization_mask(layerName,stageMasks,hiddenMasks);
        newLayer = sliced_group_normalization(oldLayer,mask);
        compressedNet = replaceLayer(compressedNet,layerName,newLayer);
        replacedNormalizations(end+1,1) = layerName; %#ok<AGROW>

    elseif isa(oldLayer,"nnet.cnn.layer.FullyConnectedLayer")
        assert(layerName == "fc","NMA:UnexpectedFullyConnected", ...
            "Unexpected fully connected layer %s.",layerName);
        newLayer = sliced_fully_connected(oldLayer,stageMasks{4});
        compressedNet = replaceLayer(compressedNet,layerName,newLayer);
    end
end

compressedNet = initialize(compressedNet);
assert(compressedNet.Initialized,"NMA:RebuildUninitialized", ...
    "The rebuilt 3-D network is unexpectedly uninitialized.");
validate_rebuilt_dimensions(compressedNet,targetWidths);

report = struct;
report.Model = spec.Model;
report.Estimator = estimator;
report.BaselineWidths = spec.BaselineWidths;
report.TargetWidths = targetWidths;
report.StageMasks = stageMasks;
report.HiddenMasks = hiddenMasks;
report.ReplacedConvolutions = replacedConvolutions;
report.ReplacedNormalizations = replacedNormalizations;
report.OriginalLearnableParameters = count_parameters(net);
report.CompressedLearnableParameters = count_parameters(compressedNet);
report.ParameterFraction = report.CompressedLearnableParameters/ ...
    report.OriginalLearnableParameters;
report.OriginalMACs = nma_resnet3d18_resource( ...
    spec.BaselineWidths,Resource="macs", ...
    NumClasses=getLayer(net,"fc").OutputSize);
report.CompressedMACs = nma_resnet3d18_resource( ...
    targetWidths,Resource="macs", ...
    NumClasses=getLayer(net,"fc").OutputSize);
report.MACFraction = report.CompressedMACs/report.OriginalMACs;
report.OriginalActivationProxy = nma_resnet3d18_resource( ...
    spec.BaselineWidths,Resource="activation", ...
    NumClasses=getLayer(net,"fc").OutputSize);
report.CompressedActivationProxy = nma_resnet3d18_resource( ...
    targetWidths,Resource="activation", ...
    NumClasses=getLayer(net,"fc").OutputSize);
report.ActivationFraction = report.CompressedActivationProxy/ ...
    report.OriginalActivationProxy;
report.CreatedAt = datetime("now",TimeZone="local");
end

function [inputMask,outputMask] = ...
    convolution_masks(name,stageMasks,hiddenMasks)
if name == "conv1"
    inputMask = 1;
    outputMask = stageMasks{1};
    return
end

token = regexp(name, ...
    "^stack([1-4]):block([12]):(conv1|conv2|skip_conv)$", ...
    "tokens","once");
assert(~isempty(token),"NMA:Unknown3DConvolution", ...
    "No 3-D ResNet-18 mask rule exists for convolution %s.",name);
stage = str2double(token{1});
block = str2double(token{2});
branch = string(token{3});

switch branch
    case "conv1"
        if stage == 1 || block == 2
            inputMask = stageMasks{stage};
        else
            inputMask = stageMasks{stage-1};
        end
        hiddenProbe = regexprep(name,"conv1$","relu1");
        outputMask = hiddenMasks.( ...
            matlab.lang.makeValidName(hiddenProbe));
    case "conv2"
        hiddenProbe = regexprep(name,"conv2$","relu1");
        inputMask = hiddenMasks.( ...
            matlab.lang.makeValidName(hiddenProbe));
        outputMask = stageMasks{stage};
    case "skip_conv"
        assert(stage > 1 && block == 1, ...
            "NMA:Unexpected3DProjection", ...
            "Projection convolution %s is not expected.",name);
        inputMask = stageMasks{stage-1};
        outputMask = stageMasks{stage};
end
end

function mask = normalization_mask(name,stageMasks,hiddenMasks)
if name == "bn1"
    mask = stageMasks{1};
    return
end
token = regexp(name, ...
    "^stack([1-4]):block([12]):(bn1|bn2|skip_bn)$", ...
    "tokens","once");
assert(~isempty(token),"NMA:Unknown3DNormalization", ...
    "No 3-D ResNet-18 mask rule exists for normalization %s.",name);
stage = str2double(token{1});
block = string(token{2});
branch = string(token{3});
if branch == "bn1"
    hiddenProbe = ...
        "stack"+stage+":block"+block+":relu1";
    mask = hiddenMasks.(matlab.lang.makeValidName(hiddenProbe));
else
    mask = stageMasks{stage};
end
end

function layer = sliced_convolution(old,inputMask,outputMask)
assert(old.PaddingValue == 0,"NMA:UnsupportedPaddingValue", ...
    "Only zero-padded 3-D convolutions are supported.");
layer = convolution3dLayer( ...
    old.FilterSize,numel(outputMask), ...
    Name=old.Name, ...
    NumChannels=numel(inputMask), ...
    Stride=old.Stride, ...
    DilationFactor=old.DilationFactor, ...
    Padding=old.PaddingSize);
layer.Weights = old.Weights(:,:,:,inputMask,outputMask);
layer.Bias = old.Bias(:,:,:,outputMask);
layer.WeightLearnRateFactor = old.WeightLearnRateFactor;
layer.WeightL2Factor = old.WeightL2Factor;
layer.BiasLearnRateFactor = old.BiasLearnRateFactor;
layer.BiasL2Factor = old.BiasL2Factor;
end

function layer = sliced_batch_normalization(old,mask)
layer = batchNormalizationLayer( ...
    Name=old.Name, ...
    Epsilon=old.Epsilon, ...
    MeanDecay=old.MeanDecay, ...
    VarianceDecay=old.VarianceDecay);
layer.Offset = old.Offset(:,:,:,mask);
layer.Scale = old.Scale(:,:,:,mask);
layer.TrainedMean = old.TrainedMean(:,:,:,mask);
layer.TrainedVariance = old.TrainedVariance(:,:,:,mask);
layer.OffsetLearnRateFactor = old.OffsetLearnRateFactor;
layer.OffsetL2Factor = old.OffsetL2Factor;
layer.ScaleLearnRateFactor = old.ScaleLearnRateFactor;
layer.ScaleL2Factor = old.ScaleL2Factor;
end

function layer = sliced_group_normalization(old,mask)
numChannels = numel(mask);
numGroups = min(double(old.NumGroups),numChannels);
while mod(numChannels,numGroups) ~= 0
    numGroups = numGroups-1;
end
layer = groupNormalizationLayer( ...
    numGroups,Name=old.Name, ...
    Epsilon=old.Epsilon);
layer.Offset = old.Offset(:,:,:,mask);
layer.Scale = old.Scale(:,:,:,mask);
layer.OffsetLearnRateFactor = old.OffsetLearnRateFactor;
layer.OffsetL2Factor = old.OffsetL2Factor;
layer.ScaleLearnRateFactor = old.ScaleLearnRateFactor;
layer.ScaleL2Factor = old.ScaleL2Factor;
end

function layer = sliced_fully_connected(old,inputMask)
layer = fullyConnectedLayer(old.OutputSize,Name=old.Name);
layer.Weights = old.Weights(:,inputMask);
layer.Bias = old.Bias;
layer.WeightLearnRateFactor = old.WeightLearnRateFactor;
layer.WeightL2Factor = old.WeightL2Factor;
layer.BiasLearnRateFactor = old.BiasLearnRateFactor;
layer.BiasL2Factor = old.BiasL2Factor;
end

function score = probe_score(calibration,probeName,estimator,expectedWidth)
row = find(calibration.Probes.Probe == probeName,1);
assert(~isempty(row),"NMA:MissingProbe", ...
    "Calibration does not contain probe %s.",probeName);
switch estimator
    case "multiplicative"
        score = calibration.Probes.MultiplicativeByChannel{row};
    case "additive"
        score = calibration.Probes.ForwardByChannel{row}+ ...
            calibration.Probes.BackwardByChannel{row};
    case "forward"
        score = calibration.Probes.ForwardByChannel{row};
    case "backward"
        score = calibration.Probes.BackwardByChannel{row};
    otherwise
        error("NMA:UnknownEstimator", ...
            "Unknown channel-score estimator %s.",estimator);
end
score = double(score(:));
assert(numel(score) == expectedWidth && all(isfinite(score)), ...
    "NMA:InvalidChannelScore", ...
    "Probe %s must provide %d finite channel scores.", ...
    probeName,expectedWidth);
end

function mask = select_channels(score,targetWidth)
if targetWidth == numel(score)
    mask = (1:numel(score)).';
    return
end
[~,order] = sort(score,"descend");
mask = sort(order(1:targetWidth));
mask = mask(:);
end

function stage = stage_index_from_name(name)
token = regexp(name,"^stack([1-4])","tokens","once");
assert(~isempty(token),"NMA:UnknownStage", ...
    "Cannot infer a 3-D ResNet stage from %s.",name);
stage = str2double(token{1});
end

function validate_rebuilt_dimensions(net,targetWidths)
assert(getLayer(net,"conv1").NumFilters == targetWidths(1), ...
    "NMA:DimensionValidationFailed","conv1 width is incorrect.");
for stage = 1:4
    for block = 1:2
        conv1 = getLayer(net, ...
            "stack"+stage+":block"+block+":conv1");
        conv2 = getLayer(net, ...
            "stack"+stage+":block"+block+":conv2");
        assert(conv1.NumFilters == targetWidths(stage) && ...
            conv2.NumFilters == targetWidths(stage), ...
            "NMA:DimensionValidationFailed", ...
            "Stack %d block %d widths are incorrect.",stage,block);
    end
end
assert(getLayer(net,"fc").InputSize == targetWidths(4), ...
    "NMA:DimensionValidationFailed","fc input width is incorrect.");
end

function count = count_parameters(net)
count = sum(cellfun(@numel,net.Learnables.Value));
end
