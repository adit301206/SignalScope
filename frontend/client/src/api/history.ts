import type { AnalysisResult } from "./analysis";

/**
 * Creates a small JPEG thumbnail for local browser history.
 *
 * We intentionally keep this small so localStorage does not fill up.
 * The original image and Grad-CAM heatmap are NOT stored.
 */
export function createHistoryThumbnail(
  file: File,
  maxSize = 240,
  quality = 0.6
): Promise<string> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const objectUrl = URL.createObjectURL(file);

    image.onload = () => {
      const scale = Math.min(
        1,
        maxSize / Math.max(image.naturalWidth, image.naturalHeight)
      );

      const canvas = document.createElement("canvas");

      canvas.width = Math.max(
        1,
        Math.round(image.naturalWidth * scale)
      );

      canvas.height = Math.max(
        1,
        Math.round(image.naturalHeight * scale)
      );

      const context = canvas.getContext("2d");

      if (!context) {
        URL.revokeObjectURL(objectUrl);
        reject(new Error("Could not create thumbnail canvas"));
        return;
      }

      context.drawImage(
        image,
        0,
        0,
        canvas.width,
        canvas.height
      );

      // JPEG keeps the thumbnail much smaller than PNG.
      const thumbnail = canvas.toDataURL(
        "image/jpeg",
        quality
      );

      URL.revokeObjectURL(objectUrl);
      resolve(thumbnail);
    };

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Could not create image thumbnail"));
    };

    image.src = objectUrl;
  });
}

export interface HistoryAttributionSummary {
  generator: string;
  confidence: number;
}

export interface HistoryProvenanceSummary {
  summary?: string;
  c2paStatus?: string;
  exifPresent?: boolean;
}

export interface HistoryEntry {
  id: string;
  createdAt: string;
  filename: string;
  label: string;
  confidence: number;
  probability_ai: number;
  probability_real: number;
  attribution?: HistoryAttributionSummary;
  provenance?: HistoryProvenanceSummary;
  fileType?: string;

  /**
   * Small local thumbnail used by the History "Re-analyze" action.
   * Optional for backward compatibility with older history entries.
   */
  thumbnail?: string;
}

const KEY = "signalscope-history";

function sanitizeEntry(raw: any): HistoryEntry {
  return {
    id: String(
      raw.id ||
        crypto.randomUUID?.() ||
        Date.now()
    ),

    createdAt:
      raw.createdAt ||
      new Date().toISOString(),

    filename:
      raw.filename ||
      "Untitled image",

    label:
      raw.label ||
      "Analysis",

    confidence:
      typeof raw.confidence === "number"
        ? raw.confidence
        : 0,

    probability_ai:
      typeof raw.probability_ai === "number"
        ? raw.probability_ai
        : 0,

    probability_real:
      typeof raw.probability_real === "number"
        ? raw.probability_real
        : 0,

    attribution: raw.attribution
      ? {
          generator: raw.attribution.generator,
          confidence: raw.attribution.confidence,
        }
      : undefined,

    provenance: raw.provenance
      ? {
          summary: raw.provenance.summary,

          c2paStatus:
            raw.provenance.c2pa?.status ||
            raw.provenance.c2paStatus,

          exifPresent:
            typeof raw.provenance.exif?.present ===
            "boolean"
              ? raw.provenance.exif.present
              : raw.provenance.exifPresent,
        }
      : undefined,

    fileType: raw.fileType,

    /**
     * IMPORTANT:
     * Keep the small thumbnail when reading history back.
     *
     * This is what allows Analyze.tsx to restore the image
     * when the user clicks "Re-analyze".
     */
    thumbnail:
      typeof raw.thumbnail === "string"
        ? raw.thumbnail
        : undefined,
  };
}

export function getHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);

    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.map(sanitizeEntry);
  } catch {
    return [];
  }
}

export function saveHistory(
  result:
    | AnalysisResult
    | Omit<HistoryEntry, "id" | "createdAt">,
  fileType?: string,
  thumbnail?: string
): HistoryEntry | null {
  try {
    const nextEntry: HistoryEntry = {
      id:
        crypto.randomUUID?.() ||
        `${Date.now()}-${Math.random()
          .toString(36)
          .substring(2, 7)}`,

      createdAt:
        new Date().toISOString(),

      filename:
        result.filename ||
        "Untitled image",

      label:
        result.label,

      confidence:
        result.confidence,

      probability_ai:
        result.probability_ai,

      probability_real:
        result.probability_real,

      attribution: result.attribution
        ? {
            generator:
              result.attribution.generator,

            confidence:
              result.attribution.confidence,
          }
        : undefined,

      provenance: result.provenance
        ? {
            summary:
              result.provenance.summary,

            c2paStatus:
              result.provenance.c2pa?.status ||
              (result.provenance as any)
                .c2paStatus,

            exifPresent:
              typeof result.provenance.exif?.present ===
              "boolean"
                ? result.provenance.exif.present
                : (result.provenance as any)
                    .exifPresent,
          }
        : undefined,

      fileType:
        fileType ||
        ("fileType" in result
          ? (result as any).fileType
          : undefined),

      /**
       * Only the small compressed thumbnail is stored.
       * Large images / heatmaps are never stored here.
       */
      thumbnail,
    };

    // Keep only the newest 12 history entries.
    let history = [
      nextEntry,
      ...getHistory(),
    ].slice(0, 12);

    try {
      localStorage.setItem(
        KEY,
        JSON.stringify(history)
      );
    } catch (quotaError) {
      console.warn(
        "localStorage quota exceeded while saving history. Trimming old history entries.",
        quotaError
      );

      /**
       * If storage is full, remove older entries
       * until the new entry can be saved.
       */
      while (history.length > 1) {
        history.pop();

        try {
          localStorage.setItem(
            KEY,
            JSON.stringify(history)
          );

          return nextEntry;
        } catch {
          // Continue trimming.
        }
      }
    }

    return nextEntry;
  } catch (err) {
    console.warn(
      "Could not save history entry:",
      err
    );

    return null;
  }
}

export function deleteHistory(id: string) {
  try {
    const updatedHistory =
      getHistory().filter(
        (entry) => entry.id !== id
      );

    localStorage.setItem(
      KEY,
      JSON.stringify(updatedHistory)
    );
  } catch {
    // Ignore storage delete errors.
  }
}

export function clearHistory() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // Ignore storage clear errors.
  }
}