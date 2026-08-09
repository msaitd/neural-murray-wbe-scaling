function [stats,loss,learnableGradients,state] = ...
    nma_probe_demand(net,X,T,probeNames,lossFcn)
%NMA_PROBE_DEMAND Collect forward, backward, and multiplicative demand.
%
% [stats,loss,learnableGradients,state] = nma_probe_demand( ...
%     net,X,T,probeNames,lossFcn)
%
% Inputs
%   net        : initialized dlnetwork
%   X          : formatted dlarray input
%   T          : targets accepted by lossFcn
%   probeNames : string array of layer names or layer output paths
%   lossFcn    : function handle, for example @(Y,T) crossentropy(Y,T)
%
% The function runs a training-mode forward pass inside dlfeval, returns the
% selected intermediate activations, and differentiates the task loss with
% respect to both learnables and those activations. It reports scalar and
% channelwise statistics for each probe:
%   ForwardMeanSquare       E[h^2]
%   BackwardMeanSquare      E[g^2]
%   MultiplicativeMeanAbs   E[|h .* g|]
%
% These are one-batch observations. A caller should aggregate them over a
% calibration stream with a prespecified mean or exponential moving average.

arguments
    net (1,1) dlnetwork
    X dlarray
    T
    probeNames string
    lossFcn (1,1) function_handle
end

probeNames = probeNames(:).';
assert(~isempty(probeNames),"NMA:NoProbes", ...
    "At least one probe layer or output path is required.");
assert(net.Initialized,"NMA:UninitializedNetwork", ...
    "Initialize the dlnetwork before collecting demand.");

[loss,learnableGradients,activations,activationGradients,state] = ...
    dlfeval(@model_loss_with_probes,net,X,T,probeNames,lossFcn);

template = struct( ...
    "Probe","", ...
    "Format","", ...
    "ForwardMeanSquare",NaN, ...
    "BackwardMeanSquare",NaN, ...
    "MultiplicativeMeanAbs",NaN, ...
    "ForwardByChannel",[], ...
    "BackwardByChannel",[], ...
    "MultiplicativeByChannel",[]);
stats = repmat(template,numel(probeNames),1);

for k = 1:numel(probeNames)
    activation = activations{k};
    gradient = activationGradients{k};
    assert(isequal(size(activation),size(gradient)), ...
        "NMA:ProbeShapeMismatch", ...
        "Activation and sensitivity shapes differ for probe %s.",probeNames(k));

    format = string(dims(activation));
    channelDimension = find(char(format) == 'C',1);
    assert(~isempty(channelDimension),"NMA:MissingChannelLabel", ...
        "Probe %s has no C-labelled dimension.",probeNames(k));

    h = extractdata(activation);
    g = extractdata(gradient);
    product = abs(h.*g);

    stats(k).Probe = probeNames(k);
    stats(k).Format = format;
    stats(k).ForwardMeanSquare = to_cpu_double(mean(h.^2,"all"));
    stats(k).BackwardMeanSquare = to_cpu_double(mean(g.^2,"all"));
    stats(k).MultiplicativeMeanAbs = to_cpu_double(mean(product,"all"));
    stats(k).ForwardByChannel = channel_reduce(h.^2,channelDimension);
    stats(k).BackwardByChannel = channel_reduce(g.^2,channelDimension);
    stats(k).MultiplicativeByChannel = ...
        channel_reduce(product,channelDimension);
end
end

function [loss,learnableGradients,activations,activationGradients,state] = ...
    model_loss_with_probes(net,X,T,probeNames,lossFcn)

taskOutput = string(net.OutputNames(1));
outputNames = [taskOutput probeNames];
rawOutputs = cell(1,numel(outputNames)+1);
[rawOutputs{:}] = forward(net,X,Outputs=outputNames);

state = rawOutputs{end};
Y = rawOutputs{1};
activations = rawOutputs(2:end-1);
loss = lossFcn(Y,T);

activationGradients = cell(size(activations));
[learnableGradients,activationGradients{:}] = ...
    dlgradient(loss,net.Learnables,activations{:});
end

function values = channel_reduce(x,channelDimension)
reduceDimensions = setdiff(1:ndims(x),channelDimension);
values = x;
for dimension = sort(reduceDimensions,"descend")
    values = mean(values,dimension);
end
values = to_cpu_double(squeeze(values));
values = values(:);
end

function value = to_cpu_double(value)
if isa(value,"gpuArray")
    value = gather(value);
end
value = double(value);
end
