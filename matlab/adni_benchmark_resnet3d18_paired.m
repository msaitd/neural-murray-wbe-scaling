function benchmark = ...
    adni_benchmark_resnet3d18_paired( ...
    networks,networkNames,X,options)
%ADNI_BENCHMARK_RESNET3D18_PAIRED Interleaved GPU latency benchmark.
%
% Every timing block executes each model once in randomized order. This
% reduces bias from thermal and clock drift. Bootstrap confidence intervals
% resample complete timing blocks and therefore preserve within-block
% pairing relative to the first (baseline) model.

arguments
    networks (1,:) cell
    networkNames (1,:) string
    X {mustBeNumeric,mustBeNonempty}
    options.WarmupIterations (1,1) double ...
        {mustBeInteger,mustBeNonnegative} = 20
    options.TimingBlocks (1,1) double ...
        {mustBeInteger,mustBePositive} = 200
    options.BootstrapReplicates (1,1) double ...
        {mustBeInteger,mustBePositive} = 5000
    options.RandomSeed (1,1) double {mustBeInteger} = 20260727
end

assert(numel(networks) >= 2 && ...
    numel(networks) == numel(networkNames), ...
    "NMA:InvalidPairedBenchmarkInputs", ...
    "Provide matching names for at least two networks.");
assert(numel(unique(networkNames)) == numel(networkNames), ...
    "NMA:DuplicateBenchmarkName", ...
    "Paired benchmark names must be unique.");
for index = 1:numel(networks)
    assert(isa(networks{index},"dlnetwork") && ...
        isscalar(networks{index}), ...
        "NMA:InvalidBenchmarkNetwork", ...
        "Every benchmark input must be a scalar dlnetwork.");
end

parallel.gpu.enableCUDAForwardCompatibility(true);
assert(canUseGPU,"NMA:GPUUnavailable", ...
    "The paired 3-D benchmark requires the configured GPU.");
device = gpuDevice;
dlX = dlarray(gpuArray(single(X)),"SSSCB");
numNetworks = numel(networks);
for index = 1:numNetworks
    for warmup = 1:options.WarmupIterations
        predict(networks{index},dlX);
        wait(device);
    end
end

rng(options.RandomSeed,"twister");
latency = zeros(options.TimingBlocks,numNetworks);
orders = zeros(options.TimingBlocks,numNetworks);
for block = 1:options.TimingBlocks
    order = randperm(numNetworks);
    orders(block,:) = order;
    for position = 1:numNetworks
        index = order(position);
        timer = tic;
        output = predict(networks{index},dlX);
        wait(device);
        latency(block,index) = toc(timer);
        outputValues = extractdata(output);
        assert(all(isfinite(outputValues),"all"), ...
            "NMA:NonfinitePairedBenchmark", ...
            "Network %s produced nonfinite output.", ...
            networkNames(index));
    end
end

medianLatency = median(latency,1).';
meanLatency = mean(latency,1).';
standardDeviation = std(latency,0,1).';
p025 = percentile(latency,2.5).';
p975 = percentile(latency,97.5).';
ratio = medianLatency/medianLatency(1);
ratioCI = zeros(numNetworks,2);
ratioCI(1,:) = [1 1];
rng(options.RandomSeed+1,"twister");
bootstrapRatio = zeros(options.BootstrapReplicates,numNetworks);
for replicate = 1:options.BootstrapReplicates
    rows = randi(options.TimingBlocks, ...
        options.TimingBlocks,1);
    medians = median(latency(rows,:),1);
    bootstrapRatio(replicate,:) = medians/medians(1);
end
for index = 2:numNetworks
    ratioCI(index,:) = percentile( ...
        bootstrapRatio(:,index),[2.5 97.5]);
end

parameterCount = zeros(numNetworks,1);
macs = zeros(numNetworks,1);
activationProxy = zeros(numNetworks,1);
widths = zeros(numNetworks,4);
for index = 1:numNetworks
    spec = nma_resnet3d18_spec(networks{index});
    widths(index,:) = spec.BaselineWidths.';
    parameterCount(index) = sum( ...
        cellfun(@numel,networks{index}.Learnables.Value));
    numClasses = getLayer(networks{index},"fc").OutputSize;
    macs(index) = nma_resnet3d18_resource( ...
        spec.BaselineWidths,Resource="macs", ...
        NumClasses=numClasses);
    activationProxy(index) = nma_resnet3d18_resource( ...
        spec.BaselineWidths,Resource="activation", ...
        NumClasses=numClasses);
end

summary = table( ...
    networkNames(:), ...
    widths(:,1),widths(:,2),widths(:,3),widths(:,4), ...
    parameterCount,parameterCount/parameterCount(1), ...
    macs,macs/macs(1), ...
    activationProxy,activationProxy/activationProxy(1), ...
    medianLatency,meanLatency,standardDeviation,p025,p975, ...
    ratio,ratioCI(:,1),ratioCI(:,2),1./medianLatency, ...
    VariableNames=[ ...
    "Network","Width1","Width2","Width3","Width4", ...
    "ParameterCount","ParameterFraction", ...
    "MACs","MACFraction", ...
    "ActivationProxy","ActivationFraction", ...
    "MedianLatencySeconds","MeanLatencySeconds", ...
    "LatencyStandardDeviation","LatencyP025","LatencyP975", ...
    "MedianLatencyRatioToBaseline", ...
    "LatencyRatioCI95Lower","LatencyRatioCI95Upper", ...
    "ThroughputVolumesPerSecond"]);

benchmark = struct;
benchmark.Summary = summary;
benchmark.RawLatencySeconds = latency;
benchmark.ExecutionOrders = orders;
benchmark.BootstrapLatencyRatio = bootstrapRatio;
benchmark.Device = string(device.Name);
benchmark.ComputeCapability = string(device.ComputeCapability);
if isprop(device,"DriverVersion")
    benchmark.DriverVersion = string(device.DriverVersion);
else
    benchmark.DriverVersion = missing;
end
benchmark.WarmupIterations = options.WarmupIterations;
benchmark.TimingBlocks = options.TimingBlocks;
benchmark.BootstrapReplicates = options.BootstrapReplicates;
benchmark.RandomSeed = options.RandomSeed;
benchmark.CreatedAt = datetime("now",TimeZone="local");
end

function values = percentile(data,percentages)
data = sort(data,1);
positions = 1+(size(data,1)-1)*(percentages(:)/100);
values = zeros(numel(percentages),size(data,2));
for index = 1:numel(percentages)
    lowerIndex = floor(positions(index));
    upperIndex = ceil(positions(index));
    weight = positions(index)-lowerIndex;
    values(index,:) = ...
        (1-weight)*data(lowerIndex,:)+ ...
        weight*data(upperIndex,:);
end
if isscalar(percentages)
    values = values(1,:);
end
end
