function training = adni_train_resnet3d18(datasetRoot,options)
%ADNI_TRAIN_RESNET3D18 Train or fine-tune the ADNI 3-D ResNet-18.

arguments
    datasetRoot (1,1) string
    options.Network = []
    options.Widths (1,4) double {mustBeInteger,mustBePositive} = ...
        [32 64 128 256]
    options.InternalNormalization (1,1) string = "batch"
    options.MaxEpochs (1,1) double {mustBeInteger,mustBePositive} = 5
    options.MiniBatchSize (1,1) double {mustBeInteger,mustBePositive} = 1
    options.InitialLearnRate (1,1) double {mustBePositive} = 1e-4
    options.L2Regularization (1,1) double {mustBeNonnegative} = 1e-4
    options.ExecutionEnvironment (1,1) string = "auto"
    options.EnableCUDAForwardCompatibility (1,1) logical = false
    options.DeterministicGPU (1,1) logical = true
    options.ValidationPatience (1,1) double ...
        {mustBeInteger,mustBePositive} = 5
    options.BatchNormalizationStatistics (1,1) string ...
        {mustBeMember(options.BatchNormalizationStatistics, ...
        ["auto","moving"])} = "auto"
    options.ResetInputNormalization (1,1) logical = true
    options.UseAugmentation (1,1) logical = false
    options.RandomSeed (1,1) double {mustBeInteger} = 20260727
    options.SaveFile (1,1) string = ""
    options.Verbose (1,1) logical = true
end

if options.EnableCUDAForwardCompatibility
    parallel.gpu.enableCUDAForwardCompatibility(true);
end
if options.ExecutionEnvironment == "gpu"
    assert(canUseGPU,"NMA:GPUUnavailable", ...
        "ExecutionEnvironment is gpu, but MATLAB cannot use the GPU.");
end
useGPU = options.ExecutionEnvironment == "gpu" || ...
    (options.ExecutionEnvironment == "auto" && canUseGPU);
if options.DeterministicGPU && useGPU
    previousDeterminism = ...
        deep.gpu.deterministicAlgorithms(true);
    previousGPUGenerator = gpurng;
    determinismCleanup = onCleanup(@() restore_gpu_state( ...
        previousDeterminism,previousGPUGenerator));
    gpurng(options.RandomSeed,"Threefry");
end

datasetManifest = fullfile(datasetRoot,"adni_3d_manifest.csv");
[trainDs,trainRows,classNames] = ...
    adni_3d_datastore( ...
    datasetManifest,"train",Augment=options.UseAugmentation);
[validationDs,validationRows,validationClasses] = ...
    adni_3d_datastore(datasetManifest,"validation");
assert(isequal(classNames,validationClasses), ...
    "NMA:SplitClassMismatch", ...
    "The train and validation splits must use identical class order.");

if isempty(options.Network)
    net = adni_resnet3d18_create( ...
        options.Widths,NumClasses=numel(classNames), ...
        InternalNormalization=options.InternalNormalization);
    initialization = "random He initialization";
else
    assert(isa(options.Network,"dlnetwork") && isscalar(options.Network), ...
        "NMA:InvalidNetwork","Network must be a scalar dlnetwork.");
    net = options.Network;
    assert(getLayer(net,"fc").OutputSize == numel(classNames), ...
        "NMA:OutputClassMismatch", ...
        "Provided network output size does not match ADNI classes.");
    initialization = "provided network";
end

classCounts = zeros(numel(classNames),1);
for classIndex = 1:numel(classNames)
    classCounts(classIndex) = sum(trainRows.group == classNames(classIndex));
end
assert(all(classCounts > 0),"NMA:MissingTrainingClass", ...
    "Every class must occur in the training split.");
classWeights = sum(classCounts)./(numel(classCounts)*classCounts);
classWeights = classWeights/mean(classWeights);
lossFunction = @(Y,T) crossentropy( ...
    Y,T,classWeights.',WeightsFormat="UC");

iterationsPerEpoch = ceil(height(trainRows)/options.MiniBatchSize);
rng(options.RandomSeed,"twister");
trainOptions = trainingOptions( ...
    "adam", ...
    InitialLearnRate=options.InitialLearnRate, ...
    MaxEpochs=options.MaxEpochs, ...
    MiniBatchSize=options.MiniBatchSize, ...
    L2Regularization=options.L2Regularization, ...
    Shuffle="every-epoch", ...
    ValidationData=validationDs, ...
    ValidationFrequency=max(1,iterationsPerEpoch), ...
    ValidationPatience=options.ValidationPatience, ...
    BatchNormalizationStatistics= ...
        options.BatchNormalizationStatistics, ...
    ResetInputNormalization=options.ResetInputNormalization, ...
    Metrics="accuracy", ...
    OutputNetwork="best-validation", ...
    ExecutionEnvironment=options.ExecutionEnvironment, ...
    Plots="none", ...
    Verbose=options.Verbose);

[trainedNet,info] = trainnet( ...
    trainDs,net,lossFunction,trainOptions);

training = struct;
training.Network = trainedNet;
training.Info = info;
training.ClassNames = classNames;
training.ClassCounts = classCounts;
training.ClassWeights = classWeights;
training.Initialization = initialization;
training.Widths = nma_resnet3d18_spec(trainedNet).BaselineWidths;
training.TrainRows = height(trainRows);
training.ValidationRows = height(validationRows);
training.TrainSubjects = numel(unique(trainRows.subject));
training.ValidationSubjects = ...
    numel(unique(validationRows.subject));
training.TestEvaluated = false;
training.Options = rmfield(options,"Network");
training.CreatedAt = datetime("now",TimeZone="local");
if strlength(options.SaveFile) > 0
    save(options.SaveFile,"training","-v7.3");
end
end

function restore_gpu_state(previousDeterminism,previousGPUGenerator)
deep.gpu.deterministicAlgorithms(previousDeterminism);
gpurng(previousGPUGenerator);
end
