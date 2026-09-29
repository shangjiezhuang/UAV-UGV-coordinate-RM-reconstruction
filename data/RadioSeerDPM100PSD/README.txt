RadioSeerDPM100PSD

Filtered 100x100 PSD dataset derived from RadioSeerDPM128PSD_4096x8.
Base crops: 1253
PSD variants: 10024
K: 30

Crop policy:
- Start from each selected 128x128 source crop.
- Search all valid 100x100 sub-crops and keep the best obstacle-aware crop.
- Require Tx local image x/y to be at least 16 pixels from every edge.
- Keep scenes whose final 100x100 building_ratio is in [0.120, 0.260].
- Require center 60x60 building_ratio in [0.080, 0.450] and inner 76x76 building_ratio >= 0.080.
- Require edge building share <= 0.400, at least 3 occupied quadrants, and at least 6 occupied 20x20 grid cells.
- Require at least 2 significant building components, at least 1 internal components, and largest component fraction <= 0.700.

Folders:
- buildings_complete: cropped building layout PNG
- building_mask: cropped building mask PNG
- non_building_mask: cropped non-building mask PNG
- antenna: cropped one-hot transmitter PNG
- gain_DPM: cropped DPM PNG
- gain_DPM_nonbuilding: cropped DPM PNG masked to non-building
- arrays_npz: cropped S/Phi/masks/images for each PSD variant
- metadata: per-PSD-variant JSON metadata

NPZ orientation notes:
- Core NPZ arrays use image y,x order, matching RadioSeerClip100:
  S, building_mask, non_building_mask, antenna, gain_DPM,
  buildings_complete, and gain_DPM_nonbuilding.
- The *_image and *_yx keys are image y,x aliases.
- The building_mask_xy and non_building_mask_xy keys are explicit x,y aliases
  for code that needs transposed masks.

Merged sources:
- RadioSeerClip100: 4 base crops converted to 32 PSD variants.

Spectrum policy:
- Phi is regenerated with center_mode=random for every row.
- Each sinc center is sampled uniformly from integer 1..K using psd_seed.
- Each row keeps its manifest n_sinc and width_min/width_max values.
- Amplitudes remain uniform in the row's amp_min/amp_max range and Phi is normalized so sum(Phi)=K.
