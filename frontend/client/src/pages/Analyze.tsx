import { useState } from "react";
import { AlertCircle, ArrowRight, Server, UploadCloud } from "lucide-react";
import { Link } from "wouter";
import { analyzeImage, type AnalysisResult } from "@/api/analysis";
import { fileToDataUrl, saveHistory } from "@/api/history";
import { AnalysisLoader, ResultView, UploadZone } from "@/components/SignalScopeUI";

type State = "empty" | "ready" | "loading" | "result" | "error";

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [state, setState] = useState<State>("empty");
  const [error, setError] = useState("");
  const chooseFile = (next: File) => { setFile(next); setResult(null); setError(""); setState("ready"); };
  const reset = () => { setFile(null); setResult(null); setError(""); setState("empty"); };
  const runAnalysis = async () => {
    if (!file) return;
    setState("loading"); setError("");
    try { const response = await analyzeImage(file); setResult(response); setState("result"); const { heatmap_image: _heatmap, ...historySafeResult } = response; saveHistory({ ...historySafeResult, thumbnail: await fileToDataUrl(file), fileType: file.type }); }
    catch (err) {
      let message = "The image could not be analyzed right now. Try again with another image.";
      if (err instanceof Error) {
        if (err.message === "backend-unavailable") {
          message = "We couldn't reach the analysis service. Check that the SignalScope backend is running and try again.";
        } else if (err.message === "file-too-large") {
          message = "The uploaded image file is too large. Please select an image under 15 MB.";
        } else if (err.message === "invalid-response") {
          message = "Received an invalid response format from the analysis service.";
        } else if (err.message) {
          message = err.message;
        }
      }
      setError(message);
      setState("error");
    }
  };
  return <div className="analyze-page"><div className="shell">
    <div className="analyze-header"><div><span className="eyebrow eyebrow--rule">ANALYSIS WORKSPACE <span>LIVE</span></span><h1>Analyze an image</h1><p>Inspect the likelihood of synthetic generation with visual evidence from the model.</p></div><div className="analysis-state"><span className={`status-dot ${state === "loading" ? "status-dot--pulse" : ""}`} />{state === "loading" ? "Analyzing" : "Model online"}</div></div>
    {state === "result" && file && result ? <ResultView result={result} file={file} onReset={reset} /> : <div className="analysis-workspace"><div className="analysis-workspace__main"><div className="workspace-label"><span>01</span><span>SELECT SOURCE</span></div><UploadZone file={file} onFile={chooseFile} onRemove={reset} disabled={state === "loading"} />{state === "loading" ? <AnalysisLoader /> : <div className="analysis-cta"><div className="analysis-cta__hint">{state === "empty" ? <><UploadCloud size={17} /> Start with one image</> : state === "error" ? <><AlertCircle size={17} /> Analysis stopped. Try again.</> : <><span className="status-dot" /> Image ready to inspect</>}</div><button className="button button--primary" disabled={!file || state === "empty"} onClick={runAnalysis}>{state === "error" ? "Try again" : "Analyze image"} <ArrowRight size={17} /></button></div>}</div><aside className="analysis-workspace__aside"><div className="aside-panel"><div className="aside-panel__header"><span className="eyebrow">BEFORE YOU BEGIN</span><span className="mono">02</span></div><h2>One image.<br /><em>A closer look.</em></h2><p>SignalScope works best when the original image is clear and has not been heavily recompressed.</p><ul><li>Accepted: JPG, PNG, WEBP, BMP</li><li>Maximum file size: 15 MB</li><li>Your image is not stored by this interface</li></ul></div><div className="aside-panel aside-panel--technical"><div className="aside-panel__header"><span className="eyebrow">MODEL STATUS</span><Server size={15} /></div><div className="model-status-row"><span>SignalScope classifier</span><span className="status-badge"><i /> Ready</span></div><div className="model-status-row"><span>Explanation layer</span><span className="status-badge"><i /> Ready</span></div></div></aside></div>}
    {error && state !== "error" ? null : error && <div className="service-error" role="alert"><AlertCircle size={17} /><div><strong>Analysis unavailable</strong><p>{error}</p></div></div>}
    <div className="workspace-footnote"><span className="mono">03 / NOTE</span><p>SignalScope provides a likelihood assessment, not definitive proof of image origin. Performance can vary across image domains and transformations.</p><Link href="/about">Learn about limitations <ArrowRight size={14} /></Link></div>
  </div></div>;
}
