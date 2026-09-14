import type { AnalysisResult } from "./analysis";

export type HistoryEntry = AnalysisResult & {
  id: string;
  createdAt: string;
  thumbnail?: string;
  fileType?: string;
};

const KEY = "signalscope-history";

export function getHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function saveHistory(entry: Omit<HistoryEntry, "id" | "createdAt">) {
  const next: HistoryEntry = { ...entry, id: crypto.randomUUID?.() || `${Date.now()}`, createdAt: new Date().toISOString() };
  const history = [next, ...getHistory()].slice(0, 12);
  localStorage.setItem(KEY, JSON.stringify(history));
  return next;
}

export function deleteHistory(id: string) {
  localStorage.setItem(KEY, JSON.stringify(getHistory().filter((entry) => entry.id !== id)));
}

export function clearHistory() {
  localStorage.removeItem(KEY);
}

export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
