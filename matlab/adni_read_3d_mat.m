function volume = adni_read_3d_mat(filePath)
%ADNI_READ_3D_MAT Read one materialized uint8 MRI tensor for a datastore.

payload = load(filePath,"volume");
assert(isfield(payload,"volume"),"NMA:Missing3DVolume", ...
    "MAT file does not contain variable 'volume': %s",filePath);
volume = payload.volume;
assert(isa(volume,"uint8") && ndims(volume) == 3, ...
    "NMA:Invalid3DVolume", ...
    "Expected a three-dimensional uint8 volume in %s.",filePath);
volume = reshape(volume,[size(volume) 1]);
end
