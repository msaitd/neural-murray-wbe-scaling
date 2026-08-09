function result = nma_allocate_single_resource(a,b,p,rho,budget,options)
%NMA_ALLOCATE_SINGLE_RESOURCE Exact Neural Murray allocation and tile repair.
%
% result = nma_allocate_single_resource(a,b,p,rho,budget)
% result = nma_allocate_single_resource(...,GroupIds=groupIds, ...)
%
% Solves the fixed-demand separable problem
%
%   minimize    sum_i a_i * w_i^(-p)
%   subject to  sum_i b_i * w_i^(rho) <= budget
%
% after exactly aggregating equality-tied edges into width groups.
% The continuous solution is globally optimal for the stated surrogate.
% Tile projection and greedy repair are feasible heuristics, not a proof of
% global mixed-integer optimality.
%
% Name-value options
%   GroupIds      : numeric group label for every edge (default: each edge)
%   Granularity   : scalar or edgewise positive integer tile (default: 1)
%   MinWidth      : scalar or edgewise positive minimum (default: tile)
%   MaxWidth      : scalar or edgewise maximum (default: Inf)
%   FillSlack     : greedily add tiles while budget permits (default: true)
%   MaxIterations : repair iteration limit (default: 100000)
%
% Output fields
%   ContinuousGroupWidths, ProjectedGroupWidths, ContinuousEdgeWidths,
%   ProjectedEdgeWidths, GroupDemand, GroupCost, ContinuousResource,
%   ProjectedResource, ContinuousDissipation, ProjectedDissipation,
%   GroupIndex, GroupLabels, Diagnostics.

arguments
    a (:,1) double
    b (:,1) double
    p (1,1) double
    rho (1,1) double
    budget (1,1) double
    options.GroupIds (:,1) double = []
    options.Granularity double = 1
    options.MinWidth double = []
    options.MaxWidth double = Inf
    options.FillSlack (1,1) logical = true
    options.MaxIterations (1,1) double = 100000
end

n = numel(a);
assert(numel(b) == n,"NMA:SizeMismatch", ...
    "a and b must contain the same number of edges.");
assert(all(isfinite(a) & a > 0),"NMA:InvalidDemand", ...
    "Every demand coefficient must be finite and positive.");
assert(all(isfinite(b) & b > 0),"NMA:InvalidCost", ...
    "Every cost coefficient must be finite and positive.");
assert(isfinite(p) && p > 0 && isfinite(rho) && rho > 0, ...
    "NMA:InvalidExponent","p and rho must be finite and positive.");
assert(isfinite(budget) && budget > 0,"NMA:InvalidBudget", ...
    "budget must be finite and positive.");
assert(isfinite(options.MaxIterations) && options.MaxIterations >= 1 && ...
    options.MaxIterations == round(options.MaxIterations), ...
    "NMA:InvalidIterationLimit", ...
    "MaxIterations must be a positive integer.");

if isempty(options.GroupIds)
    groupLabels = (1:n).';
else
    assert(numel(options.GroupIds) == n && all(isfinite(options.GroupIds)), ...
        "NMA:InvalidGroups","GroupIds must supply one finite label per edge.");
    groupLabels = options.GroupIds;
end

[uniqueLabels,~,groupIndex] = unique(groupLabels,"stable");
m = numel(uniqueLabels);

A = accumarray(groupIndex,a,[m 1],@sum);
B = accumarray(groupIndex,b,[m 1],@sum);

edgeTiles = expand_edge_option(options.Granularity,n,"Granularity");
assert(all(edgeTiles == round(edgeTiles)),"NMA:NonIntegerTile", ...
    "Granularity values must be positive integers.");
groupTiles = zeros(m,1);
for g = 1:m
    values = edgeTiles(groupIndex == g);
    tile = values(1);
    for k = 2:numel(values)
        tile = lcm(tile,values(k));
    end
    groupTiles(g) = tile;
end

if isempty(options.MinWidth)
    edgeMin = edgeTiles;
else
    edgeMin = expand_edge_option(options.MinWidth,n,"MinWidth");
end
edgeMax = expand_edge_option(options.MaxWidth,n,"MaxWidth",true);

groupMin = accumarray(groupIndex,edgeMin,[m 1],@max);
groupMax = accumarray(groupIndex,edgeMax,[m 1],@min);
groupMin = groupTiles .* ceil(groupMin ./ groupTiles);
finiteMax = isfinite(groupMax);
groupMax(finiteMax) = groupTiles(finiteMax) .* ...
    floor(groupMax(finiteMax) ./ groupTiles(finiteMax));
assert(all(groupMax >= groupMin),"NMA:IncompatibleBounds", ...
    "At least one quotient group has MaxWidth below its tiled MinWidth.");

alpha = 1/(p+rho);
S = sum(B .* (A./B).^(rho*alpha));
scale = (budget/S)^(1/rho);
uContinuous = scale .* (A./B).^alpha;

continuousResource = resource_value(uContinuous,B,rho);
continuousDissipation = dissipation_value(uContinuous,A,p);

uProjected = groupTiles .* floor(uContinuous ./ groupTiles);
uProjected = max(uProjected,groupMin);
uProjected = min(uProjected,groupMax);

minimumResource = resource_value(groupMin,B,rho);
assert(minimumResource <= budget*(1+1e-12),"NMA:InfeasibleMinimum", ...
    "The budget cannot support the requested minimum tiled widths.");

removedTiles = 0;
addedTiles = 0;
iterations = 0;

% If minima or maxima made the initial projection infeasible, remove the
% tile with the smallest surrogate penalty per unit of resource recovered.
while resource_value(uProjected,B,rho) > budget*(1+1e-12)
    iterations = iterations + 1;
    assert(iterations <= options.MaxIterations,"NMA:RepairLimit", ...
        "Budget-removal repair exceeded MaxIterations.");

    canRemove = (uProjected-groupTiles) >= groupMin;
    assert(any(canRemove),"NMA:RepairFailed", ...
        "No removable tile remains although the projected model is over budget.");

    candidate = find(canRemove);
    oldWidth = uProjected(candidate);
    newWidth = oldWidth-groupTiles(candidate);
    penalty = A(candidate) .* (newWidth.^(-p)-oldWidth.^(-p));
    saved = B(candidate) .* (oldWidth.^rho-newWidth.^rho);
    score = penalty ./ saved;
    [~,localIndex] = min(score);
    chosen = candidate(localIndex);
    uProjected(chosen) = uProjected(chosen)-groupTiles(chosen);
    removedTiles = removedTiles+1;
end

% Fill remaining slack with the feasible tile that gives the largest
% surrogate decrease per incremental resource.
if options.FillSlack
    while true
        iterations = iterations + 1;
        assert(iterations <= options.MaxIterations,"NMA:RepairLimit", ...
            "Slack-filling repair exceeded MaxIterations.");

        currentResource = resource_value(uProjected,B,rho);
        candidateWidth = uProjected+groupTiles;
        deltaResource = B .* (candidateWidth.^rho-uProjected.^rho);
        canAdd = candidateWidth <= groupMax & ...
            currentResource+deltaResource <= budget*(1+1e-12);
        if ~any(canAdd)
            break
        end

        benefit = A .* (uProjected.^(-p)-candidateWidth.^(-p));
        score = benefit ./ deltaResource;
        score(~canAdd) = -Inf;
        [~,chosen] = max(score);
        uProjected(chosen) = candidateWidth(chosen);
        addedTiles = addedTiles+1;
    end
end

projectedResource = resource_value(uProjected,B,rho);
projectedDissipation = dissipation_value(uProjected,A,p);
assert(projectedResource <= budget*(1+1e-10),"NMA:BudgetViolation", ...
    "Internal error: projected allocation exceeds the budget.");

result = struct;
result.ContinuousGroupWidths = uContinuous;
result.ProjectedGroupWidths = uProjected;
result.ContinuousEdgeWidths = uContinuous(groupIndex);
result.ProjectedEdgeWidths = uProjected(groupIndex);
result.GroupDemand = A;
result.GroupCost = B;
result.GroupGranularity = groupTiles;
result.GroupMinWidth = groupMin;
result.GroupMaxWidth = groupMax;
result.GroupIndex = groupIndex;
result.GroupLabels = uniqueLabels;
result.ContinuousResource = continuousResource;
result.ProjectedResource = projectedResource;
result.Budget = budget;
result.ContinuousDissipation = continuousDissipation;
result.ProjectedDissipation = projectedDissipation;
result.Diagnostics = struct( ...
    "RemovedTiles",removedTiles, ...
    "AddedTiles",addedTiles, ...
    "RepairIterations",iterations, ...
    "BudgetSlack",budget-projectedResource, ...
    "ProjectedToContinuousDissipationRatio", ...
        projectedDissipation/continuousDissipation);
end

function values = expand_edge_option(value,n,name,allowInf)
if nargin < 4
    allowInf = false;
end
if isscalar(value)
    values = repmat(value,n,1);
else
    values = value(:);
    assert(numel(values) == n,"NMA:InvalidOptionSize", ...
        "%s must be scalar or contain one value per edge.",name);
end
if allowInf
    valid = values > 0 & ~isnan(values);
    qualification = " (Inf is allowed)";
else
    valid = isfinite(values) & values > 0;
    qualification = "";
end
assert(all(valid),"NMA:InvalidOption", ...
    "%s values must be positive%s.",name,qualification);
end

function value = resource_value(widths,cost,rho)
value = sum(cost .* widths.^rho);
end

function value = dissipation_value(widths,demand,p)
value = sum(demand .* widths.^(-p));
end
