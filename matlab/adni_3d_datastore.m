function [ds,rows,classNames] = ...
    adni_3d_datastore(datasetManifest,split,options)
%ADNI_3D_DATASTORE Create a combined predictor-label datastore.

arguments
    datasetManifest (1,1) string
    split (1,1) string
    options.Augment (1,1) logical = false
end

manifest = readtable( ...
    datasetManifest,Delimiter=",",TextType="string", ...
    VariableNamingRule="preserve");
rows = manifest(manifest.split == split,:);
assert(height(rows) > 0,"NMA:Empty3DSplit", ...
    "No 3-D rows found for split %s.",split);
assert(all(isfile(rows.matPath)),"NMA:MissingPrepared3DVolume", ...
    "At least one prepared 3-D volume is missing.");

classNames = ["AD";"CN";"MCI"];
labels = categorical(rows.group,classNames);
assert(~any(isundefined(labels)),"NMA:UnexpectedADNIClasses", ...
    "Expected only AD, CN, and MCI labels.");

if options.Augment
    readFunction = @adni_read_3d_augmented;
else
    readFunction = @adni_read_3d_mat;
end
predictors = fileDatastore( ...
    cellstr(rows.matPath), ...
    ReadFcn=readFunction, ...
    FileExtensions=".mat");
responses = arrayDatastore(labels,IterationDimension=1);
ds = combine(predictors,responses);
end
