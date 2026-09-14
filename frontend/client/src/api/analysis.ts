export interface GeneratorCandidate {
  generator: string;
  confidence: number;
}

export interface GeneratorAttribution {
  generator: string;
  confidence: number;
  top_2?: GeneratorCandidate[];
}

export interface AnalysisResult {
  label: string;
  confidence: number;
  probability_ai: number;
  probability_real: number;
  filename: string;
  message: string;
  heatmap_image: string;
  attribution?: GeneratorAttribution;
}

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function analyzeImage(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;

  try {
    response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("backend-unavailable");
  }

  if (!response.ok) {
    if (response.status === 413) {
      throw new Error("file-too-large");
    }

    let detail = "";

    try {
      const errorData = await response.json();
      detail = errorData?.detail || "";
    } catch {
      // Ignore JSON parsing errors.
    }

    throw new Error(
      detail || `Prediction request failed with status ${response.status}`
    );
  }

  const data = await response.json();

  if (
    typeof data.label !== "string" ||
    typeof data.confidence !== "number" ||
    typeof data.probability_ai !== "number" ||
    typeof data.probability_real !== "number"
  ) {
    throw new Error("invalid-response");
  }

  return {
    label: data.label,
    confidence: data.confidence,
    probability_ai: data.probability_ai,
    probability_real: data.probability_real,
    filename: data.filename || file.name,
    message:
      data.message ||
      "This is a likelihood assessment based on the model's learned visual patterns.",
    heatmap_image: data.heatmap_image
      ? data.heatmap_image.startsWith("data:")
        ? data.heatmap_image
        : `data:image/png;base64,${data.heatmap_image}`
      : "",
    attribution: data.attribution,
  };
}