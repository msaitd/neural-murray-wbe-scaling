function allocation = ...
    nma_resnet3d18_allocate_multiresource(calibration,options)
%NMA_RESNET3D18_ALLOCATE_MULTIRESOURCE Allocate widths under exact resources.
%
% The fixed-demand continuous problem is solved in log-width:
%
%   minimize sum_s a_s * w_s^(-p)
%   subject to C_j(w) <= budget_j for every requested resource j.
%
% C_j is the exact positive coupled 3-D ResNet-18 resource polynomial used
% elsewhere in this project. Downward tile projection preserves feasibility.
% Feasible slack is filled using continuous KKT shadow prices. The continuous
% solution is globally optimal for the fixed-demand log-convex problem; the
% tiled solution is a deterministic feasible heuristic.

arguments
    calibration (1,1) struct
    options.Resources (1,:) string = ["parameters","macs"]
    options.BudgetFractions (1,:) double {mustBePositive} = [0.75 0.75]
    options.Estimator (1,1) string = "multiplicative"
    options.NumClasses (1,1) double {mustBeInteger,mustBePositive} = 3
    options.P (1,1) double {mustBePositive} = 2
    options.Granularity double = []
    options.MinWidth double = []
    options.MaxWidth double = []
    options.FillSlack (1,1) logical = true
    options.MaxFillIterations (1,1) double ...
        {mustBeInteger,mustBePositive} = 100000
end

assert(isfield(calibration,"Spec") && ...
    isfield(calibration,"StageDemand"), ...
    "NMA:InvalidCalibration", ...
    "Use nma_resnet3d18_calibrate to construct calibration.");
spec = calibration.Spec;
baselineWidths = double(spec.BaselineWidths(:));

resources = lower(string(options.Resources(:)));
budgetFractions = double(options.BudgetFractions(:));
assert(numel(resources) >= 2 && ...
    numel(resources) == numel(budgetFractions), ...
    "NMA:InvalidMultiResourceOptions", ...
    "Supply at least two resources and one budget fraction per resource.");
assert(numel(unique(resources)) == numel(resources) && ...
    all(ismember(resources, ...
    ["parameters","macs","activation"])), ...
    "NMA:InvalidMultiResourceNames", ...
    "Resources must be unique parameters, macs, or activation entries.");
assert(all(budgetFractions <= 1), ...
    "NMA:InvalidMultiResourceBudget", ...
    "Every budget fraction must be in (0,1].");

estimator = lower(options.Estimator);
switch estimator
    case "multiplicative"
        observedDemand = calibration.StageDemand.MultiplicativeMeanAbs;
    case "additive"
        observedDemand = ...
            calibration.StageDemand.ForwardMeanSquare + ...
            calibration.StageDemand.BackwardMeanSquare;
    case "forward"
        observedDemand = calibration.StageDemand.ForwardMeanSquare;
    case "backward"
        observedDemand = calibration.StageDemand.BackwardMeanSquare;
    otherwise
        error("NMA:UnknownEstimator", ...
            "Unknown demand estimator %s.",estimator);
end
observedDemand = max(double(observedDemand(:)),eps);
demandCoefficient = observedDemand.*baselineWidths.^options.P;
objectiveScale = sum(observedDemand);

granularity = expand_four(option_or_default( ...
    options.Granularity,spec.DefaultGranularity),"Granularity");
minWidth = expand_four(option_or_default( ...
    options.MinWidth,spec.DefaultMinWidths),"MinWidth");
maxWidth = expand_four(option_or_default( ...
    options.MaxWidth,spec.DefaultMaxWidths),"MaxWidth");
assert(all(granularity == round(granularity)), ...
    "NMA:InvalidGranularity","Granularity must contain integers.");
minWidth = granularity.*ceil(minWidth./granularity);
maxWidth = granularity.*floor(maxWidth./granularity);
assert(all(maxWidth >= minWidth), ...
    "NMA:IncompatibleBounds", ...
    "At least one tiled MaxWidth is below MinWidth.");

numResources = numel(resources);
baselineResources = zeros(numResources,1);
budgets = zeros(numResources,1);
minimumResources = zeros(numResources,1);
for resourceIndex = 1:numResources
    baselineResources(resourceIndex) = resource_value( ...
        baselineWidths,resources(resourceIndex),options.NumClasses);
    budgets(resourceIndex) = ...
        budgetFractions(resourceIndex)*baselineResources(resourceIndex);
    minimumResources(resourceIndex) = resource_value( ...
        minWidth,resources(resourceIndex),options.NumClasses);
end
assert(all(minimumResources <= budgets*(1+1e-12)), ...
    "NMA:InfeasibleMultiResourceMinimum", ...
    "The requested resource budgets cannot support minimum widths.");

initialWidths = feasible_initial_point( ...
    minWidth,baselineWidths,resources,budgets,options.NumClasses);
objective = @(logWidths) sum( ...
    demandCoefficient.*exp(-options.P*logWidths))/objectiveScale;
constraint = @(logWidths) multiresource_constraint( ...
    logWidths,resources,budgets,options.NumClasses);
solverOptions = optimoptions( ...
    "fmincon", ...
    Algorithm="interior-point", ...
    Display="none", ...
    ConstraintTolerance=1e-9, ...
    OptimalityTolerance=1e-9, ...
    StepTolerance=1e-11, ...
    MaxFunctionEvaluations=20000, ...
    MaxIterations=3000);

[logContinuous,~,exitFlag,solverOutput,lambda] = fmincon( ...
    objective,log(initialWidths),[],[],[],[], ...
    log(minWidth),log(maxWidth),constraint,solverOptions);
continuousWidths = exp(logContinuous);
continuousResources = all_resource_values( ...
    continuousWidths,resources,options.NumClasses);
continuousDissipation = dissipation( ...
    continuousWidths,demandCoefficient,options.P);
assert(exitFlag > 0,"NMA:MultiResourceSolverFailed", ...
    "Continuous multi-resource optimization failed: %s", ...
    solverOutput.message);
assert(all(continuousResources <= budgets*(1+1e-7)), ...
    "NMA:MultiResourceContinuousBudgetViolation", ...
    "The continuous solution exceeds a resource budget.");

projectedWidths = granularity.*floor( ...
    continuousWidths./granularity);
projectedWidths = max(projectedWidths,minWidth);
projectedWidths = min(projectedWidths,maxWidth);
projectedResources = all_resource_values( ...
    projectedWidths,resources,options.NumClasses);
assert(all(projectedResources <= budgets*(1+1e-10)), ...
    "NMA:MultiResourceProjectionViolation", ...
    "Downward tile projection unexpectedly exceeds a budget.");

shadowPrices = max(double(lambda.ineqnonlin(:)),0);
if numel(shadowPrices) ~= numResources || ...
        all(shadowPrices <= eps)
    shadowPrices = ones(numResources,1);
end
shadowPrices = shadowPrices/sum(shadowPrices);
fillIterations = 0;
if options.FillSlack
    while true
        fillIterations = fillIterations+1;
        assert(fillIterations <= options.MaxFillIterations, ...
            "NMA:MultiResourceFillLimit", ...
            "Multi-resource slack filling exceeded its iteration limit.");
        currentDissipation = dissipation( ...
            projectedWidths,demandCoefficient,options.P);
        scores = -inf(4,1);
        trials = cell(4,1);
        trialResourceValues = cell(4,1);
        for stage = 1:4
            if projectedWidths(stage)+granularity(stage) > ...
                    maxWidth(stage)
                continue
            end
            trial = projectedWidths;
            trial(stage) = trial(stage)+granularity(stage);
            trialResources = all_resource_values( ...
                trial,resources,options.NumClasses);
            if any(trialResources > budgets*(1+1e-12))
                continue
            end
            benefit = currentDissipation-dissipation( ...
                trial,demandCoefficient,options.P);
            normalizedIncrease = ...
                (trialResources-projectedResources)./budgets;
            effectiveCost = sum(shadowPrices.*normalizedIncrease);
            if effectiveCost <= eps
                effectiveCost = max(normalizedIncrease);
            end
            scores(stage) = benefit/max(effectiveCost,eps);
            trials{stage} = trial;
            trialResourceValues{stage} = trialResources;
        end
        [bestScore,chosen] = max(scores);
        if ~isfinite(bestScore)
            fillIterations = fillIterations-1;
            break
        end
        projectedWidths = trials{chosen};
        projectedResources = trialResourceValues{chosen};
    end
end

projectedDissipation = dissipation( ...
    projectedWidths,demandCoefficient,options.P);
assert(all(projectedResources <= budgets*(1+1e-10)), ...
    "NMA:MultiResourceProjectedBudgetViolation", ...
    "The tiled allocation exceeds a resource budget.");

resourceTable = table( ...
    resources,baselineResources,budgetFractions,budgets, ...
    minimumResources,continuousResources,projectedResources, ...
    projectedResources./baselineResources, ...
    budgets-projectedResources,shadowPrices, ...
    VariableNames=[ ...
    "Resource","Baseline","BudgetFraction","Budget", ...
    "Minimum","Continuous","Projected","ProjectedFraction", ...
    "BudgetSlack","ContinuousShadowPrice"]);

allocation = struct;
allocation.Model = spec.Model;
allocation.Resource = "multiple";
allocation.Resources = resources;
allocation.ResourceTable = resourceTable;
allocation.Estimator = estimator;
allocation.NumClasses = options.NumClasses;
allocation.P = options.P;
allocation.BaselineWidths = baselineWidths;
allocation.ObservedDemand = observedDemand;
allocation.DemandCoefficient = demandCoefficient;
allocation.BudgetFractions = budgetFractions;
allocation.BaselineResources = baselineResources;
allocation.Budgets = budgets;
allocation.MinimumResources = minimumResources;
allocation.ContinuousWidths = continuousWidths;
allocation.ContinuousResources = continuousResources;
allocation.ContinuousDissipation = continuousDissipation;
allocation.TargetWidths = projectedWidths;
allocation.ProjectedResources = projectedResources;
allocation.ResourceFractions = ...
    projectedResources./baselineResources;
allocation.ProjectedDissipation = projectedDissipation;
allocation.ExitFlag = exitFlag;
allocation.SolverOutput = solverOutput;
allocation.ShadowPrices = shadowPrices;
allocation.Diagnostics = struct( ...
    "FillIterations",fillIterations, ...
    "BudgetSlack",budgets-projectedResources, ...
    "ProjectedToContinuousDissipationRatio", ...
    projectedDissipation/continuousDissipation);
allocation.CreatedAt = datetime("now",TimeZone="local");
end

function [c,ceq] = multiresource_constraint( ...
    logWidths,resources,budgets,numClasses)
values = all_resource_values( ...
    exp(logWidths),resources,numClasses);
c = values./budgets-1;
ceq = [];
end

function widths = feasible_initial_point( ...
    minimum,baseline,resources,budgets,numClasses)
low = 0;
high = 1;
for iteration = 1:80
    fraction = 0.5*(low+high);
    candidate = minimum+fraction*(baseline-minimum);
    values = all_resource_values( ...
        candidate,resources,numClasses);
    if all(values <= budgets)
        low = fraction;
    else
        high = fraction;
    end
end
widths = minimum+low*(baseline-minimum);
end

function values = all_resource_values(widths,resources,numClasses)
values = zeros(numel(resources),1);
for index = 1:numel(resources)
    values(index) = resource_value( ...
        widths,resources(index),numClasses);
end
end

function value = resource_value(widths,resource,numClasses)
value = nma_resnet3d18_resource( ...
    widths,Resource=resource,NumClasses=numClasses);
end

function value = dissipation(widths,demand,p)
value = sum(demand.*widths.^(-p));
end

function value = option_or_default(value,defaultValue)
if isempty(value)
    value = defaultValue;
end
end

function values = expand_four(value,name)
if isscalar(value)
    values = repmat(double(value),4,1);
else
    values = double(value(:));
end
assert(numel(values) == 4 && ...
    all(isfinite(values) & values > 0), ...
    "NMA:InvalidMultiResourceOption", ...
    "%s must be a positive scalar or four-vector.",name);
end

