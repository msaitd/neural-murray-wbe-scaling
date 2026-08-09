function volume = adni_read_3d_augmented(filePath)
%ADNI_READ_3D_AUGMENTED Apply label-preserving 3-D training perturbations.

volume = single(adni_read_3d_mat(filePath))/255;
if rand < 0.5
    volume = flip(volume,2);
end
gamma = 0.9+0.2*rand;
volume = volume.^gamma;
gain = 0.95+0.1*rand;
volume = gain*volume+0.01*randn(size(volume),"single");
volume = min(max(volume,0),1);
volume = uint8(round(255*volume));
end
