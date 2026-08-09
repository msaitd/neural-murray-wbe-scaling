function net = adni_resnet3d18_create(widths,options)
%ADNI_RESNET3D18_CREATE Create the fixed ADNI 3-D ResNet-18 contract.

arguments
    widths (1,4) double {mustBeInteger,mustBePositive} = ...
        [32 64 128 256]
    options.InputSize (1,4) double {mustBeInteger,mustBePositive} = ...
        [96 96 96 1]
    options.NumClasses (1,1) double {mustBeInteger,mustBePositive} = 3
    options.Normalization (1,1) string = "zscore"
    options.InternalNormalization (1,1) string = "batch"
end

assert(all(diff(widths) >= 0),"NMA:NonMonotone3DWidths", ...
    "3-D ResNet stage widths must be nondecreasing.");
net = resnet3dNetwork( ...
    options.InputSize,options.NumClasses, ...
    StackDepth=[2 2 2 2], ...
    NumFilters=widths, ...
    InitialNumFilters=widths(1), ...
    BottleneckType="none", ...
    Normalization=options.Normalization);

internalNormalization = lower(options.InternalNormalization);
assert(ismember(internalNormalization,["batch","group"]), ...
    "NMA:Unknown3DNormalization", ...
    "InternalNormalization must be batch or group.");
if internalNormalization == "group"
    for layerIndex = 1:numel(net.Layers)
        oldLayer = net.Layers(layerIndex);
        if ~isa(oldLayer,"nnet.cnn.layer.BatchNormalizationLayer")
            continue
        end
        numChannels = numel(oldLayer.Scale);
        numGroups = min(8,numChannels);
        while mod(numChannels,numGroups) ~= 0
            numGroups = numGroups-1;
        end
        newLayer = groupNormalizationLayer( ...
            numGroups,Name=oldLayer.Name, ...
            Epsilon=oldLayer.Epsilon);
        newLayer.Offset = oldLayer.Offset;
        newLayer.Scale = oldLayer.Scale;
        net = replaceLayer(net,oldLayer.Name,newLayer);
    end
    net = initialize(net);
end
end
