# Data access and non-redistribution boundary

The associated experiment used controlled-access MRI images and metadata from the Alzheimer's Disease
Neuroimaging Initiative (ADNI). This repository intentionally contains source code and protocol text
only.

Researchers seeking to reproduce the participant-level analysis must:

1. obtain independent authorization from ADNI/LONI;
2. accept and follow the current ADNI Data Use Agreement;
3. download the authorized data through the official archive;
4. keep images, metadata, derived manifests, predictions, checkpoints, trained weights, and logs outside
   Git version control; and
5. follow the current ADNI publication and Data and Publications Committee review process.

Never attach controlled data or participant identifiers to a GitHub issue, pull request, discussion,
release, or support request. The `.gitignore` file blocks common research-data formats as a secondary
safeguard, but access compliance remains the responsibility of each authorized investigator.
