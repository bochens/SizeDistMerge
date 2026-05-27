# Next Release Plan: PUTLS-MERGE Fallback When FIMS Is Not Available

## Scope

This release adds PUTLS-MERGE support to the 5-minute and 1-minute merged size-distribution production workflow when FIMS is not available.

Temporal regularization is intentionally out of scope for this release.

## Goal

Keep the current FIMS-based merge as the primary production path.

When FIMS is missing or fails availability/QC for a chunk, use PUTLS-MERGE as the low-size reference so the merge can still produce a flagged output instead of dropping the chunk.

## Key Rule

FIMS remains the trusted reference whenever available.

PUTLS-MERGE is only used as a fallback reference when FIMS is not available for the chunk.

## Inputs

Existing inputs remain:

- APS
- UHSAS
- POPS
- FIMS
- Inlet flag
- Microphysical/CPC data

New input:

- PUTLS-MERGE size distribution product

## Fallback Decision Logic

For each production chunk:

1. Try the existing FIMS-based merge first.
2. If FIMS is available and passes chunk QC, use the normal FIMS workflow.
3. If FIMS is missing or fails required chunk-level availability/QC, try PUTLS-MERGE fallback.
4. If PUTLS-MERGE is also unavailable or invalid, skip the chunk with a clear log message.

This should be chunk-level logic, not only whole-day logic.

## PUTLS-MERGE Reference Use

For fallback chunks:

- Use PUTLS-MERGE as the reference distribution for the lower-size region.
- Use only complete PUTLS-MERGE bins up to the chosen cutoff.
- Candidate cutoff: bins with upper edge `<= 300 nm`.
- Record the effective cutoff from the actual selected bin edges.

The cutoff should be selected from the actual PUTLS-MERGE bin metadata, not hard-coded blindly.

## Alignment And Fitting Behavior

For FIMS chunks:

- Keep the existing four-instrument workflow.
- FIMS is the reference.
- Fit UHSAS refractive index.
- Fit POPS refractive index.
- Fit APS density.
- Build the merged consensus product.

For PUTLS-MERGE fallback chunks:

- PUTLS-MERGE replaces FIMS as the low-size reference.
- APS density should still be fit normally because APS sizing depends strongly on density.
- UHSAS and POPS fitting can still run, but the fitted refractive indices should be treated as weaker/diagnostic because PUTLS-MERGE may already include UHSAS/POPS information.
- The output must clearly flag that the reference was PUTLS-MERGE, not FIMS.

## Output Flags

Add flags to the NetCDF output:

```text
reference_source_flag
0 = FIMS
1 = PUTLS_MERGE
```

```text
ri_fit_quality_flag
0 = trusted_FIMS_RI
1 = weak_PUTLS_RI
```

Recommended additional output variables:

```text
reference_aligned_dNdlogDp
putls_ref_aligned_dNdlogDp
```

For FIMS chunks:

- `reference_source_flag = 0`
- `ri_fit_quality_flag = 0`
- `reference_aligned_dNdlogDp` contains the selected/aligned FIMS reference

For PUTLS fallback chunks:

- `reference_source_flag = 1`
- `ri_fit_quality_flag = 1`
- `reference_aligned_dNdlogDp` contains the selected/aligned PUTLS-MERGE reference
- `putls_ref_aligned_dNdlogDp` also stores the PUTLS-MERGE reference explicitly

## Implementation Pieces

Add or update only narrow parts of the production code:

1. Add a PUTLS-MERGE reader.
2. Add reference-source selection logic.
3. Generalize FIMS-specific reference assumptions so the reference can be either `FIMS` or `PUTLS_MERGE`.
4. Add source/quality flags to NetCDF writing.
5. Make ICT conversion preserve the new flags.
6. Add clear logging for every chunk:
   - FIMS merge used
   - PUTLS fallback used
   - skipped because neither reference was usable

## Tests Needed

Minimum tests for this release:

- PUTLS-MERGE reader parses bin edges and distributions correctly.
- Bin cutoff selection chooses only complete bins up to the cutoff.
- FIMS-available chunks still use FIMS.
- FIMS-missing chunks use PUTLS-MERGE.
- Output flags are correct.
- Existing FIMS-only output is unchanged when PUTLS fallback is disabled.
- NetCDF to ICT conversion preserves the new flags.

## Release Acceptance Criteria

The release is ready when:

- Existing FIMS-based production still works.
- FIMS chunks are not changed except for added metadata/flags.
- PUTLS-MERGE fallback chunks produce merged distributions instead of being dropped.
- PUTLS fallback chunks are clearly identifiable in NetCDF and ICT outputs.
- APS density fitting still runs for PUTLS fallback chunks.
- UHSAS/POPS RI values from PUTLS fallback chunks are marked as weak/diagnostic.

