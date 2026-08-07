# AHEPA / ds004504 — request for MRI-derived table

Send as email (English). Fill bracketed fields before sending.

**To:** tzallas@uoi.gr  
**Cc:** a.miltiadous@uoi.gr, ktzimourta@uowm.gr  
**Optional Cc (if imaging sits with neurology):** afrantou@gmail.com, ioannidispanosgr@yahoo.gr, ngrigoriadis@auth.gr  
**Subject:** Request for MRI-derived summary table linked to OpenNeuro ds004504 (AD/FTD/CN)

---

Dear Professor Tzallas, Dr Miltiadous, and Dr Tzimourta,

I am writing about your openly released AHEPA EEG resource:

Miltiadous et al., *Data* 2023, 8, 95.  
OpenNeuro **ds004504** (and the complementary eyes-open set **ds006036**).

We have been using the public EEG + MMSE / age / sex / Group labels under the dataset licence. Our analysis treats the 10–20 montage as a diffusion-style spatial probe and asks whether a subject-level readout remains associated with MMSE **after** classical EEG spectral features (in particular θ/α) and demographics are removed. To test whether that leftover tracks **brain structure** rather than only the EEG voltmeter, we would be very grateful for a **de-identified numeric table** on the **same 88 participants**, keyed by the OpenNeuro `participant_id` (`sub-001` …).

We are **not** requesting raw DICOM/NIfTI, clinical notes, or any directly identifying information.

## Minimum useful table (CSV or TSV)

One row per `participant_id`, if available from routine clinical MRI closest in time to the EEG:

| column | notes |
|--------|--------|
| `participant_id` | exact ds004504 ID |
| `mri_available` | yes / no / unknown |
| `mri_eeg_interval_days` | signed days, MRI minus EEG; blank if unknown |
| `field_strength_T` | e.g. 1.5 / 3 |
| `icv_mm3` | intracranial volume |
| `hippocampus_L_mm3`, `hippocampus_R_mm3` | any standard pipeline (FreeSurfer / FIRST / clinical report) |
| `mean_thickness_lh_mm`, `mean_thickness_rh_mm` | global mean cortical thickness |
| `entorhinal_thickness_mm` and/or `temporal_thickness_mm` | if already extracted |
| `pipeline` + `software_version` | one string is enough |

If only a subset of rows or only hippocampal volumes exist, that is still highly useful. A simple “MRI not acquired / not retained” flag per ID would also help us avoid over-interpreting missingness.

## Optional (only if already on file)

- Estimated Total Intracranial Volume–normalised hippocampal volumes  
- Clinical visual atrophy scores (e.g. MTA / Koedam / GCA), if coded  
- Scanner model  

We do **not** need DTI, fMRI, or full surface meshes for this request.

## Use, governance, and credit

- Purpose: same-subject test of EEG probe leftover vs structural atrophy, in AD / FTD / CN.  
- Outputs would be aggregate statistics and figures only; no attempt at re-identification.  
- We will cite the *Data* descriptor and OpenNeuro accession in any manuscript.  
- We are happy to sign a short data-use agreement, share a one-page analysis plan, and **offer co-authorship** if you consider the table a substantial contribution (or acknowledgement only, if you prefer).  
- If MRI exists but cannot leave AHEPA, we can instead send a locked analysis script for you to run locally and return only summary coefficients.

If this is better directed to the 2nd Department of Neurology at AHEPA, I would be grateful if you could forward this message or suggest the appropriate contact.

Thank you for releasing a carefully documented clinical EEG set — it has already been valuable. I appreciate any guidance, even if the imaging cannot be shared.

Kind regards,

**[Full name]**  
**[Position, group]**  
**[Institution, city, country]**  
**[Institutional email]**  
**[ORCID, optional]**  
**[One-sentence lab description, optional]**

---

## Short follow-up (if no reply in 10–14 days)

Subject: Re: Request for MRI-derived summary table linked to ds004504

Dear Professor Tzallas,

I am writing briefly to follow up on my email of **[date]** regarding a de-identified MRI summary table keyed to ds004504 `participant_id`. Even a yes/no on whether clinical T1 (or hippocampal volumes) exist for this cohort would help us plan. Happy to adjust the request to whatever is easiest on your side.

Thank you again,  
**[Name]**

---

## Notes for you (do not send)

- Corresponding author on the descriptor: **Alexandros T. Tzallas** `tzallas@uoi.gr`.  
- First author: **Andreas Miltiadous** `a.miltiadous@uoi.gr`.  
- EEG co-author: **Katerina D. Tzimourta** `ktzimourta@uowm.gr` (also listed as `ktzimourta@uoi.gr` on some pages).  
- Imaging, if it exists, likely sits with AHEPA neurology (Afrantou / Ioannidis / Grigoriadis), not only the Ioannina informatics group — that is why the optional Cc and the “please forward” line matter.  
- Keep the first email short when pasting into a mail client: you can drop the markdown table and keep the bullet list.  
- Do not attach unpublished HNF/PNF manuscripts in the first contact.  
- If they ask what leftover is: “subject-level EEG spatial-diffusion readout residualized on age, sex, and θ/α power.”
