# One-page detailed PDF report

From the result page, select **Save as PDF** and choose your browser's PDF printer. This produces a one-page A4 report with:

- Original image and model attention overlay when Grad-CAM is available
- Assessment, classifier score and two-class distribution (when the stage-one gate passes)
- DINOv2 open-set gate cosine distance and actual threshold (for new scans)
- Existing client-side image suitability measurements, with status and flagged guidance
- Peak relative attention, high-attention coverage and interpretation caveats
- Validation performance metrics only when real figures are supplied and the model matches
- Tailored recommended next steps and a screening-only disclaimer

**Evaluation note:** `static/js/model-info.js` currently marks the evaluation confusion matrix as sample data via `isPlaceholder: true`. This PDF intentionally omits its sample accuracy/F1/recall figures. Enter verified measurements and change the flag only once those values are supported by real evaluation. They must match the deployed model.

**Older scan note:** previously saved results will not have the new gate-distance metadata, so the PDF will correctly say that the numeric distance wasn't recorded for that scan. Scan again to include it.

The print function uses the browser's print dialog, requests A4 portrait with small margins, and triggers the print dialog synchronously on click. The server-produced overlay is available without waiting for canvas export; an optional refreshed overlay is prepared separately and never blocks printing. The on-screen design and existing camera flow are unchanged.
