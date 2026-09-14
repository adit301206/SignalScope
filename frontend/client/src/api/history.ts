import type { AnalysisResult } from "./analysis";

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
  thumbnail?: string; // Kept optional for backward compatibility
}

const KEY = "signalscope-history";

function sanitizeEntry(raw: any): HistoryEntry {
  return {
    id: String(raw.id || crypto.randomUUID?.() || Date.now()),
    createdAt: raw.createdAt || new Date().toISOString(),
    filename: raw.filename || "Untitled image",
    label: raw.label || "Analysis",
    confidence: typeof raw.confidence === "number" ? raw.confidence : 0,
    probability_ai: typeof raw.probability_ai === "number" ? raw.probability_ai : 0,
    probability_real: typeof raw.probability_real === "number" ? raw.probability_real : 0,
    attribution: raw.attribution
      ? {
          generator: raw.attribution.generator,
          confidence: raw.attribution.confidence,
        }
      : undefined,
    provenance: raw.provenance
      ? {
          summary: raw.provenance.summary,
          c2paStatus: raw.provenance.c2pa?.status || raw.provenance.c2paStatus,
          exifPresent:
            typeof raw.provenance.exif?.present === "boolean"
              ? raw.provenance.exif.present
              : raw.provenance.exifPresent,
        }
      : undefined,
    fileType: raw.fileType,
    // Heatmap images and large raw base64 images are intentionally stripped
  };
}

export function getHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map(sanitizeEntry);
  } catch {
    return [];
  }
}

export function saveHistory(
  result: AnalysisResult | Omit<HistoryEntry, "id" | "createdAt">,
  fileType?: string
): HistoryEntry | null {
  try {
    const nextEntry: HistoryEntry = {
      id: crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      createdAt: new Date().toISOString(),
      filename: result.filename || "Untitled image",
      label: result.label,
      confidence: result.confidence,
      probability_ai: result.probability_ai,
      probability_real: result.probability_real,
      attribution: result.attribution
        ? {
            generator: result.attribution.generator,
            confidence: result.attribution.confidence,
          }
        : undefined,
      provenance: result.provenance
        ? {
            summary: result.provenance.summary,
            c2paStatus: result.provenance.c2pa?.status || (result.provenance as any).c2paStatus,
            exifPresent:
              typeof result.provenance.exif?.present === "boolean"
                ? result.provenance.exif.present
                : (result.provenance as any).exifPresent,
          }
        : undefined,
      fileType: fileType || ("fileType" in result ? (result as any).fileType : undefined),
    };

    let history = [nextEntry, ...getHistory()].slice(0, 12);

    try {
      localStorage.setItem(KEY, JSON.stringify(history));
    } catch (quotaError) {
      console.warn("localStorage quota exceeded while saving history. Trimming old history entries.", quotaError);
      while (history.length > 1) {
        history.pop();
        try {
          localStorage.setItem(KEY, JSON.stringify(history));
          return nextEntry;
        } catch {
          // Continue trimming
        }
      }
    }

    return nextEntry;
  } catch (err) {
    console.warn("Could not save history entry:", err);
    return null;
  }
}

export function deleteHistory(id: string) {
  try {
    localStorage.setItem(KEY, JSON.stringify(getHistory().filter((entry) => entry.id !== id)));
  } catch {
    // Ignore storage delete errors
  }
}

export function clearHistory() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // Ignore storage clear errors
  }
}
