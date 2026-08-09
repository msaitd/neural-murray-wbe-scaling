function calibration = ...
    nma_resnet3d18_calibrate(net,volumeFiles,targets,options)
%NMA_RESNET3D18_CALIBRATE Measure 3-D activation-gradient transport demand.

arguments
    net (1,1) dlnetwork
    volumeFiles string
    targets {mustBeNumeric}
    options.MiniBatchSize (1,1) double {mustBeInteger,mustBePositive} = 1
    options.MaxBatches (1,1) double {mustBePositive} = Inf
    options.UseGPU (1,1) logical = false
    options.ClassWeights (1,:) double {mustBePositive} = [1 1 1]
    options.Verbose (1,1) logical = true
    options.ProgressFrequency (1,1) double ...
        {mustBeInteger,mustBePositive} = 1
end

spec = nma_resnet3d18_spec(net);
volumeFiles = volumeFiles(:);
assert(~isempty(volumeFiles) && all(isfile(volumeFiles)), ...
    "NMA:Missing3DCalibrationFiles", ...
    "Every 3-D calibration file must exist.");
assert(isvector(targets) && numel(targets) == numel(volumeFiles), ...
    "NMA:TargetCountMismatch", ...
    "Supply one class index for every 3-D calibration volume.");
targets = double(targets(:));
numClasses = getLayer(net,"fc").OutputSize;
assert(numel(options.ClassWeights) == numClasses, ...
    "NMA:ClassWeightMismatch", ...
    "ClassWeights must contain one value per output class.");
assert(all(targets >= 1 & targets <= numClasses & ...
    targets == round(targets)),"NMA:InvalidClassIndex", ...
    "Target indices must be valid one-based class integers.");

numBatchesAvailable = ceil(numel(volumeFiles)/options.MiniBatchSize);
numBatches = min(numBatchesAvailable,options.MaxBatches);
assert(isinf(options.MaxBatches) || ...
    options.MaxBatches == round(options.MaxBatches), ...
    "NMA:InvalidMaxBatches","MaxBatches must be an integer or Inf.");

probeNames = spec.AllProbeNames;
numProbes = numel(probeNames);
forwardSum = zeros(numProbes,1);
backwardSum = zeros(numProbes,1);
multiplicativeSum = zeros(numProbes,1);
forwardChannels = cell(numProbes,1);
backwardChannels = cell(numProbes,1);
multiplicativeChannels = cell(numProbes,1);
losses = zeros(numBatches,1);

classWeights = single(options.ClassWeights(:).');
lossFunction = @(Y,T) crossentropy( ...
    Y,T,classWeights,WeightsFormat="UC");

for batch = 1:numBatches
    first = (batch-1)*options.MiniBatchSize+1;
    last = min(batch*options.MiniBatchSize,numel(volumeFiles));
    indices = first:last;
    xb = zeros([spec.InputSize numel(indices)],"single");
    for sample = 1:numel(indices)
        xb(:,:,:,:,sample) = ...
            single(adni_read_3d_mat(volumeFiles(indices(sample))));
    end
    if options.UseGPU
        xb = gpuArray(xb);
    end
    dlX = dlarray(xb,"SSSCB");

    oneHot = zeros(numClasses,numel(indices),"single");
    linear = sub2ind(size(oneHot), ...
        reshape(targets(indices),1,[]),1:numel(indices));
    oneHot(linear) = 1;
    if options.UseGPU
        oneHot = gpuArray(oneHot);
    end
    dlT = dlarray(oneHot,"CB");

    [batchStats,loss] = nma_probe_demand( ...
        net,dlX,dlT,probeNames,lossFunction);
    losses(batch) = cpu_double(extractdata(loss));
    for probe = 1:numProbes
        forwardSum(probe) = forwardSum(probe)+ ...
            batchStats(probe).ForwardMeanSquare;
        backwardSum(probe) = backwardSum(probe)+ ...
            batchStats(probe).BackwardMeanSquare;
        multiplicativeSum(probe) = multiplicativeSum(probe)+ ...
            batchStats(probe).MultiplicativeMeanAbs;
        forwardChannels{probe} = add_vector( ...
            forwardChannels{probe},batchStats(probe).ForwardByChannel);
        backwardChannels{probe} = add_vector( ...
            backwardChannels{probe},batchStats(probe).BackwardByChannel);
        multiplicativeChannels{probe} = add_vector( ...
            multiplicativeChannels{probe}, ...
            batchStats(probe).MultiplicativeByChannel);
    end

    if options.Verbose && ...
            (batch == 1 || ...
            mod(batch,options.ProgressFrequency) == 0 || ...
            batch == numBatches)
        fprintf("3-D NMA calibration batch %d/%d, loss %.6g\n", ...
            batch,numBatches,losses(batch));
    end
end

forwardMean = forwardSum/numBatches;
backwardMean = backwardSum/numBatches;
multiplicativeMean = multiplicativeSum/numBatches;
for probe = 1:numProbes
    forwardChannels{probe} = forwardChannels{probe}/numBatches;
    backwardChannels{probe} = backwardChannels{probe}/numBatches;
    multiplicativeChannels{probe} = ...
        multiplicativeChannels{probe}/numBatches;
end

probeTable = table( ...
    probeNames,forwardMean,backwardMean,multiplicativeMean, ...
    forwardChannels,backwardChannels,multiplicativeChannels, ...
    VariableNames=[ ...
        "Probe","ForwardMeanSquare","BackwardMeanSquare", ...
        "MultiplicativeMeanAbs","ForwardByChannel", ...
        "BackwardByChannel","MultiplicativeByChannel"]);

calibration = struct;
calibration.Model = spec.Model;
calibration.Spec = spec;
calibration.Probes = probeTable;
calibration.StageDemand = probeTable(1:4,:);
calibration.HiddenDemand = probeTable(5:end,:);
calibration.Losses = losses;
calibration.MeanLoss = mean(losses);
calibration.TargetMode = "provided";
calibration.NumImages = min( ...
    numel(volumeFiles),numBatches*options.MiniBatchSize);
calibration.NumBatches = numBatches;
calibration.MiniBatchSize = options.MiniBatchSize;
calibration.UseGPU = options.UseGPU;
calibration.ClassWeights = double(classWeights);
calibration.CreatedAt = datetime("now",TimeZone="local");
end

function total = add_vector(total,value)
value = double(value(:));
if isempty(total)
    total = zeros(size(value));
else
    assert(isequal(size(total),size(value)), ...
        "NMA:InconsistentProbeWidth", ...
        "A probe changed channel count between calibration batches.");
end
total = total+value;
end

function value = cpu_double(value)
if isa(value,"gpuArray")
    value = gather(value);
end
value = double(value);
end
