# Problem Statement

Generative image tools make it easier to create convincing but synthetic imagery. That can complicate source verification for misinformation, fraud, product listings, and other contexts where a photograph may be trusted more than an AI-generated image.

SignalScope addresses a narrower question: based on learned visual patterns, is an input image more likely real or AI-generated? The project reports a likelihood rather than an accusation. It cannot establish provenance, identify who created an image, verify a real-world event, or determine whether every generated/edited image is synthetic.

Generalisation is a central challenge: a detector can score well on the generators represented in training and fail on another image distribution. The repository therefore includes a recorded held-out Defactify evaluation and degradation tests. These results are useful evidence, but do not prove performance on all generators or real-world sources.
