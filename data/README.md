# Dataset card

## Files and spatial unit

| Tile | Width (pixels) | Height (pixels) | JSON polygon count |
| --- | ---: | ---: | ---: |
| 4_5 | 22320 | 14677 | 4 |
| 4_6 | 22321 | 14677 | 66 |
| 4_7 | 22320 | 14678 | 54 |
| 4_8 | 22321 | 14678 | 5 |

Each tile has a TIFF image, an indexed PNG annotation mask and a JSON annotation file. The JSON files include embedded image data. Four adjacent tiles form one scene. In this release, “original” means the supplied source files preserved byte for byte; it does not mean that these orthomosaic tiles are untouched camera exposures. The acquisition date, sensor and actual flight height are not independently verified.

## Annotation meaning

The native PNGs use indexed labels 0/1. Read the palette indices directly when inspecting these labels; converting an indexed image to grayscale can change its numeric values. The polygons outline coarse Solanum rostratum patches, including some intervening soil. They do not provide exact leaf boundaries, individual-plant counts or confirmed absence outside the annotated patches. There are 129 polygon entries in total, which should not be interpreted as 129 independent plants.

## Download and verification

Download all four ZIP assets from the v1.0.0 release. The dataset manifest records each original file's size and SHA-256 and each ZIP's size, SHA-256 and download URL. An additional SHA256.json is included inside each ZIP. Original bytes were retained without redrawing annotations, resampling images or generating synthetic replacements during publication.

## Use and limitations

Use spatial separation when designing experiments; random window splits can share nearby image content. Splits reported in the associated study are within this single scene. Repeated model fits and image-detail settings do not create additional independent fields. Any new benchmark should state its split coordinates, buffer handling and excluded positions explicitly. Historical split and missingness information is documented in ../provenance/RESEARCH_SCOPE.md.

## Rights

The declaring rights holder authorized full-image and annotation download and reuse under CC BY 4.0. Credit the dataset, version, repository and license, and indicate modifications. See LICENSE.md. These permissions apply to these local files, not to external datasets referenced by the code.
