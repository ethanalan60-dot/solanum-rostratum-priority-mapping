# Provenance and interpretation

Four adjacent georeferenced tiles (4_5, 4_6, 4_7, 4_8) cover one scene. They are not four independent fields, flights or ecological replicates. Their acquisition date, camera and actual flight height have not been independently established. File editing metadata is not acquisition evidence.

The release contains the supplied TIFF, indexed PNG and JSON bytes without image editing. JSON polygon counts are 4, 66, 54 and 5 (129 total); annotations describe coarse patches that can include soil. They are not exact leaf masks or plant counts. Unlabelled pixels do not prove weed absence. Hashes establish file identity, not the complete history before files were supplied. No generative-image use was identified in the reviewed research records; original camera-level authenticity is not certified by this repository.

Original preparation deduplicated 2,008 records to 1,979 coordinates. After exclusion of 299 buffer positions, the retained train/validation/evaluation counts were 985/357/338. Nine eligible positions were omitted in the historical preparation (one positive buffer position and eight unlabelled positions outside the buffer); the outside-buffer eligible set of 1,688 was not fully analysed. Do not present this release as a complete independent-field validation. Of 338 evaluation windows, 284 were unlabelled.

Image-detail settings downsample 256-pixel windows to 256, 128, 64 or 42 pixels and restore them to 256 pixels. They simulate loss of detail at fixed extent, not actual new flights. The grid-origin extension retains all 4,608 combinations: 24 models x 4 detail settings x 3 grid sizes x 4 origins x 4 area limits. Shared-scene model settings are not biological replicates. No measured control efficacy, yield gain or economic benefit is claimed.

## AI assistance
OpenAI Codex assisted coding, computational workflows, record organisation and manuscript drafting/revision. This was broader than language polishing alone. Scientific data and outcomes must be assessed from their underlying files and computation; AI assistance does not provide independent scientific verification. Public packaging removes private machine paths and unrelated administrative files, while retaining this disclosure and original-source hash mappings. Historical private records remain unchanged. No claim is made that the code or prose was entirely human-authored.

Evaluation labels were decoded/cached in original preparation. Later input restrictions do not make the entire study label-blind or preregistered. Grid-origin sensitivity is a post-development analysis of saved predictions, not a new independent field experiment.

WeedsGalore is an external dataset (https://github.com/GFZ/weedsgalore), governed by its original terms. Its imagery is not redistributed under the local dataset license. External-model fits do not demonstrate transfer of the local model to independent fields.
