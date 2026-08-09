function report = adni_prepare_3d_dataset(splitManifest,outputRoot,options)
%ADNI_PREPARE_3D_DATASET Materialize subject-safe fixed-size ADNI volumes.

arguments
    splitManifest (1,1) string = ""
    outputRoot (1,1) string = ""
    options.SubjectIds string = strings(0,1)
    options.MaxSubjects (1,1) double {mustBePositive} = Inf
    options.OutputSize (1,3) double {mustBeInteger,mustBePositive} = ...
        [96 96 96]
    options.Overwrite (1,1) logical = false
    options.Verbose (1,1) logical = true
end

projectRoot = fileparts(fileparts(mfilename("fullpath")));
if strlength(splitManifest) == 0
    splitManifest = fullfile( ...
        projectRoot,"adni_work","adni_baseline_subject_split.csv");
end
if strlength(outputRoot) == 0
    outputRoot = fullfile(projectRoot,"adni_work","adni_3d");
end

source = readtable( ...
    splitManifest,Delimiter=",",TextType="string", ...
    VariableNamingRule="preserve");
required = [ ...
    "Image Data ID","Subject","Group","Visit", ...
    "NIfTI Path","Split","Class Index"];
assert(all(ismember(required,string(source.Properties.VariableNames))), ...
    "NMA:InvalidSplitManifest", ...
    "The subject split manifest is missing required columns.");

if ~isempty(options.SubjectIds)
    source = source(ismember(source.Subject,options.SubjectIds),:);
end
if isfinite(options.MaxSubjects)
    source = source(1:min(height(source),options.MaxSubjects),:);
end
assert(height(source) > 0,"NMA:NoSelectedSubjects", ...
    "No subjects were selected for 3-D preparation.");

numSubjects = height(source);
subject = source.Subject;
imageDataId = source.("Image Data ID");
group = source.Group;
classIndex = source.("Class Index");
split = source.Split;
niftiPath = source.("NIfTI Path");
matPath = strings(numSubjects,1);
intensityLower = zeros(numSubjects,1);
intensityUpper = zeros(numSubjects,1);
nonzeroFraction = zeros(numSubjects,1);
originalSize = strings(numSubjects,1);
reorientedSize = strings(numSubjects,1);
croppedSize = strings(numSubjects,1);
written = false(numSubjects,1);

for row = 1:numSubjects
    destination = fullfile(outputRoot,split(row),group(row));
    if ~isfolder(destination)
        mkdir(destination);
    end
    fileName = sprintf( ...
        "%s_%s_3d.mat",subject(row),imageDataId(row));
    filePath = fullfile(destination,fileName);

    [volume,metadata] = adni_extract_3d( ...
        niftiPath(row),OutputSize=options.OutputSize);
    if options.Overwrite || ~isfile(filePath)
        save(filePath,"volume","-v7");
        written(row) = true;
    end

    matPath(row) = filePath;
    intensityLower(row) = metadata.IntensityLower;
    intensityUpper(row) = metadata.IntensityUpper;
    nonzeroFraction(row) = metadata.NonzeroFraction;
    originalSize(row) = size_string(metadata.OriginalSize);
    reorientedSize(row) = size_string(metadata.ReorientedSize);
    croppedSize(row) = size_string(metadata.CroppedSize);

    if options.Verbose && (mod(row,10) == 0 || row == numSubjects)
        fprintf("ADNI 3-D subject %d/%d\n",row,numSubjects);
    end
end

datasetManifest = table( ...
    subject,imageDataId,group,classIndex,split,niftiPath,matPath, ...
    intensityLower,intensityUpper,nonzeroFraction, ...
    originalSize,reorientedSize,croppedSize,written);
manifestPath = fullfile(outputRoot,"adni_3d_manifest.csv");
writetable(datasetManifest,manifestPath,Encoding="UTF-8");

report = struct;
report.NumSubjects = numSubjects;
report.NumVolumes = height(datasetManifest);
report.NumWritten = sum(written);
report.OutputSize = options.OutputSize;
report.OutputRoot = outputRoot;
report.ManifestPath = manifestPath;
report.SplitCounts = groupcounts(categorical(split));
report.SplitLabels = categories(categorical(split));
report.GroupCounts = groupcounts(categorical(group));
report.GroupLabels = categories(categorical(group));
report.CreatedAt = datetime("now",TimeZone="local");
disp(report)
end

function value = size_string(dimensions)
value = strjoin(string(dimensions),"x");
end
