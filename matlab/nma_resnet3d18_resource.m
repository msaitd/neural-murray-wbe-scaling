function value = nma_resnet3d18_resource(widths,options)
%NMA_RESNET3D18_RESOURCE Exact analytical cost of the fixed 96-cube model.

arguments
    widths (4,1) double {mustBePositive}
    options.Resource (1,1) string = "parameters"
    options.NumClasses (1,1) double {mustBeInteger,mustBePositive} = 3
end

w1 = widths(1);
w2 = widths(2);
w3 = widths(3);
w4 = widths(4);
resource = lower(options.Resource);
switch resource
    case "parameters"
        value = ...
            343*w1 + 3*w1 + ...
            108*w1^2 + 12*w1 + ...
            stage_parameters(w1,w2) + ...
            stage_parameters(w2,w3) + ...
            stage_parameters(w3,w4) + ...
            options.NumClasses*w4 + options.NumClasses;
    case {"macs","flops"}
        spatial = [24;12;6;3];
        value = ...
            48^3*343*w1 + ...
            spatial(1)^3*108*w1^2 + ...
            spatial(2)^3*stage_macs(w1,w2) + ...
            spatial(3)^3*stage_macs(w2,w3) + ...
            spatial(4)^3*stage_macs(w3,w4) + ...
            options.NumClasses*w4;
    case "activation"
        activationCost = 5*[24;12;6;3].^3;
        value = sum(activationCost.*widths);
    otherwise
        error("NMA:UnknownCoupledResource", ...
            "Resource must be parameters, macs, flops, or activation.");
end
end

function value = stage_parameters(previous,current)
value = 28*previous*current + 81*current^2 + 15*current;
end

function value = stage_macs(previous,current)
value = 28*previous*current + 81*current^2;
end
