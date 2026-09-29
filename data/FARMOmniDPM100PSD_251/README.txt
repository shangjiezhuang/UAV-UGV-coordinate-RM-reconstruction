FARMOmniDPM100PSD

Small 100x100 PSD dataset converted from jliang097/FARM_training_test.
Base crops: 251
PSD variants: 2008
K: 30

Source policy:
- Use FARM iso omni antenna maps only.
- Use one selected transmitter/crop per FARM scene.
- Selected scenes are D1/scene_4, D1/scene_7, and D1/scene_9.
- The spatial map S is derived from FARM layer index 4 and normalized per crop.

Folders match RadioSeerDPM100PSD:
- buildings_complete, building_mask, non_building_mask, antenna
- gain_DPM, gain_DPM_nonbuilding
- arrays_npz and metadata

NPZ orientation notes:
- Core NPZ arrays use image y,x order.
- building_mask_xy and non_building_mask_xy are explicit x,y aliases.

Spectrum policy:
- Phi is regenerated for every PSD variant.
- Sinc centers are sampled uniformly from integer 1..K.
- Phi is normalized so sum(Phi)=K.
