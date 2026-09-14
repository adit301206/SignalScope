# SignalScope Demo Script

1. **Problem (20 seconds):** Explain that visually convincing synthetic images can make source assessment difficult. SignalScope offers a likelihood assessment, not proof of origin.
2. **Start services (10 seconds):** Show the FastAPI service and the frontend running locally. Mention that the interface needs the backend to analyse uploads.
3. **Upload and detect (45 seconds):** On Analyze, upload a JPEG/PNG/WebP/BMP smaller than 15 MB. Run analysis and show the likely-real/AI label plus both probabilities.
4. **Explain (35 seconds):** Toggle the original, heatmap, and overlay views. State precisely that Grad-CAM visualises model attention for the selected class; it does not prove a particular artifact.
5. **Attribution, if configured (25 seconds):** If the attribution checkpoint is intentionally available, show its top-two generator classes and label it experimental. If not, say the API gracefully omits it because the checkpoint is not bundled.
6. **Evidence (40 seconds):** Open Insights and explain the recorded 600-image Defactify mixed-model result: ROC-AUC 0.8788, macro-F1 0.7303, and 27% FPR. Emphasize that the run is held out by image but not strict generator-held-out for the mixed model.
7. **Robustness (35 seconds):** Open Robustness. Show that quality-50 JPEG was tested, and that resize/blur increased false positives in the recorded run.
8. **Close responsibly (20 seconds):** State the limitations: no provenance/C2PA, no calibration, no adversarial tests, and no definitive proof. Use the system alongside source context and verified provenance where possible.

No demo video or screenshots are committed. Do not add a demo-video link until one is available.
