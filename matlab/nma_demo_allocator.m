%% Neural Murray Allocator: non-empirical synthetic smoke example
% This script checks the analytic allocator and graph-group projection only.
% It is not an experiment on a neural network and is not paper evidence.

clear
clc

% Six graph edges. Edges 2 and 3 are tied by a residual addition, and
% edges 4 and 5 are tied by another compatibility constraint.
a = [1.0; 4.0; 3.0; 8.0; 6.0; 2.0];
b = [1.0; 1.2; 1.1; 0.8; 0.9; 1.4];
groupIds = [1; 2; 2; 3; 3; 4];

p = 2;
rho = 1;
budget = 192;

result = nma_allocate_single_resource( ...
    a,b,p,rho,budget, ...
    GroupIds=groupIds, ...
    Granularity=8, ...
    MinWidth=8, ...
    FillSlack=true);

disp("Continuous group widths")
disp(result.ContinuousGroupWidths)
disp("Projected group widths")
disp(result.ProjectedGroupWidths)
disp("Projected edge widths")
disp(result.ProjectedEdgeWidths)
disp(result.Diagnostics)

assert(result.ProjectedResource <= budget*(1+1e-10))
assert(result.ProjectedEdgeWidths(2) == result.ProjectedEdgeWidths(3))
assert(result.ProjectedEdgeWidths(4) == result.ProjectedEdgeWidths(5))
