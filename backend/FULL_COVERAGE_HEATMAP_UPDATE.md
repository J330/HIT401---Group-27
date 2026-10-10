# Full-coverage heatmap + simplified result actions (2026-10-10)

- Removed **Copy summary** from the results page and its clipboard handler.
- Retained **Save as PDF** and **Check another leaf**.
- For scans with successful background removal, every LEAF pixel now has visible colour while the removed background is transparent:
  blue = lowest relative attribution, cyan/green = intermediate, yellow/red = highest.
- The API-generated overlay (also used as the initial PDF fallback) and the interactive
  browser canvas use the same leaf-masked palette and opacity formula. The on-screen
  opacity slider still adjusts strength.
- This is **relative predicted-class attribution**, not a mask of image areas the
  model ignored, and not a confirmed disease/lesion outline.
- No changes to disease predictions, learned weights, or input images.

Replace the old `backend/` directory contents with the updated files (preserve any
separately stored model checkpoints, configuration/secrets and database). Restart
`python manage.py runserver`, refresh the browser, and **run a new scan**.
The session's old heatmap may still hold the previous colour data until replaced.
