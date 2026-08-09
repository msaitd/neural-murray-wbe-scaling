function [volume,metadata] = adni_extract_3d(niftiPath,options)
%ADNI_EXTRACT_3D Convert one ADNI T1 NIfTI scan to a fixed 3-D tensor.
%
% The scan is reoriented to MATLAB's canonical transverse convention,
% robustly intensity clipped, cropped to a foreground bounding box, and
% resized to a fixed output grid. The returned tensor is uint8 so the
% materialized data set remains compact; network-side z-score normalization
% is calculated from the training split only by trainnet.

arguments
    niftiPath (1,1) string
    options.OutputSize (1,3) double {mustBeInteger,mustBePositive} = ...
        [96 96 96]
    options.LowerPercentile (1,1) double = 0.5
    options.UpperPercentile (1,1) double = 99.5
    options.ForegroundFraction (1,1) double = 0.02
    options.CropMarginFraction (1,1) double = 0.05
end

assert(isfile(niftiPath),"NMA:NIfTINotFound", ...
    "NIfTI file does not exist: %s",niftiPath);
assert(options.LowerPercentile >= 0 && ...
    options.UpperPercentile <= 100 && ...
    options.LowerPercentile < options.UpperPercentile, ...
    "NMA:InvalidPercentiles","Invalid intensity percentile interval.");
assert(options.ForegroundFraction >= 0 && ...
    options.ForegroundFraction < 1, ...
    "NMA:InvalidForegroundFraction", ...
    "ForegroundFraction must be in [0,1).");
assert(options.CropMarginFraction >= 0 && ...
    options.CropMarginFraction <= 0.5, ...
    "NMA:InvalidCropMargin", ...
    "CropMarginFraction must be in [0,0.5].");

medical = medicalVolume(niftiPath);
originalSize = size(medical.Voxels);
originalOrientation = string(medical.Orientation);
originalMeaning = string(medical.DataDimensionMeaning);

if medical.Orientation ~= "transverse" || ...
        ~isequal(medical.DataDimensionMeaning, ...
        ["posterior" "left" "inferior"])
    medical = updateOrientation(medical,"transverse");
end

voxels = single(medical.Voxels);
voxels(~isfinite(voxels)) = 0;
nonzero = voxels(voxels ~= 0);
assert(~isempty(nonzero),"NMA:EmptyVolume", ...
    "The NIfTI volume contains no nonzero finite voxels: %s",niftiPath);

lower = vector_percentile(nonzero,options.LowerPercentile);
upper = vector_percentile(nonzero,options.UpperPercentile);
assert(isfinite(lower) && isfinite(upper) && upper > lower, ...
    "NMA:DegenerateIntensityRange", ...
    "Robust intensity range is degenerate for %s.",niftiPath);

normalized = (voxels-lower)/(upper-lower);
normalized = min(max(normalized,0),1);
foreground = normalized > options.ForegroundFraction;
[rowRange,columnRange,sliceRange] = ...
    foreground_bounding_box(foreground,options.CropMarginFraction);
cropped = normalized(rowRange,columnRange,sliceRange);

resized = imresize3( ...
    cropped,options.OutputSize,Method="linear",Antialiasing=true);
resized = min(max(resized,0),1);
volume = uint8(round(255*resized));
assert(isequal(size(volume),options.OutputSize), ...
    "NMA:Unexpected3DOutputSize", ...
    "The 3-D output does not have the requested dimensions.");
assert(any(volume(:) > 0),"NMA:Empty3DOutput", ...
    "The prepared 3-D tensor is empty.");

metadata = struct;
metadata.NIfTIPath = niftiPath;
metadata.OriginalSize = originalSize;
metadata.OriginalOrientation = originalOrientation;
metadata.OriginalDataDimensionMeaning = originalMeaning;
metadata.ReorientedSize = size(voxels);
metadata.ReorientedOrientation = string(medical.Orientation);
metadata.ReorientedDataDimensionMeaning = ...
    string(medical.DataDimensionMeaning);
metadata.VoxelSpacing = medical.VoxelSpacing;
metadata.IntensityLower = lower;
metadata.IntensityUpper = upper;
metadata.ForegroundBoundingBox = [ ...
    rowRange(1) rowRange(end); ...
    columnRange(1) columnRange(end); ...
    sliceRange(1) sliceRange(end)];
metadata.CroppedSize = size(cropped);
metadata.OutputSize = options.OutputSize;
metadata.NonzeroFraction = nnz(volume)/numel(volume);
end

function value = vector_percentile(values,percent)
values = sort(double(values(:)));
position = 1+(numel(values)-1)*(percent/100);
lowerIndex = floor(position);
upperIndex = ceil(position);
if lowerIndex == upperIndex
    value = values(lowerIndex);
else
    weight = position-lowerIndex;
    value = (1-weight)*values(lowerIndex)+weight*values(upperIndex);
end
end

function [rowRange,columnRange,sliceRange] = ...
    foreground_bounding_box(mask,marginFraction)
rows = find(squeeze(any(any(mask,2),3)));
columns = find(squeeze(any(any(mask,1),3)));
slices = find(squeeze(any(any(mask,1),2)));
assert(~isempty(rows) && ~isempty(columns) && ~isempty(slices), ...
    "NMA:EmptyForeground","No foreground voxels survived thresholding.");
rowRange = expand_range(rows,size(mask,1),marginFraction);
columnRange = expand_range(columns,size(mask,2),marginFraction);
sliceRange = expand_range(slices,size(mask,3),marginFraction);
end

function indices = expand_range(active,limit,marginFraction)
first = active(1);
last = active(end);
margin = round((last-first+1)*marginFraction);
first = max(1,first-margin);
last = min(limit,last+margin);
indices = first:last;
end
