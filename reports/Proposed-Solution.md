# Proposed Solution

The implemented core is a two-class EfficientNet-B0 image classifier. At inference, an image is converted to RGB, resized to 224 × 224, converted to a tensor, and normalized with ImageNet mean and standard deviation. Softmax produces real and AI-generated probabilities; a 0.5 AI probability threshold determines the label.

The FastAPI endpoint also generates a Grad-CAM overlay for the class selected by the binary classifier. When a compatible external attribution checkpoint is configured, it runs a separate EfficientNet-B0 with six classes and returns its top two scores.

Not implemented in the current repository: temperature or other probability calibration, generated textual explanations, EXIF/C2PA checks, image-text consistency scoring, and active/adversarial defence. The related terms in earlier planning material describe intended scope, not current functionality.
