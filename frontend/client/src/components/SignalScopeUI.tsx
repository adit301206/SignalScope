import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "wouter";
import {
  AlertTriangle,
  ArrowRight,
  Camera,
  Check,
  ChevronDown,
  FileImage,
  Info,
  LoaderCircle,
  Menu,
  Moon,
  ScanLine,
  ShieldCheck,
  Sun,
  Upload,
  X,
} from "lucide-react";
import type { AnalysisResult, GeneratorAttribution, GeneratorCandidate, Provenance } from "@/api/analysis";
import { useTheme } from "@/contexts/ThemeContext";

export function BrandMark() {
  return (
    <Link href="/" className="brand-mark" aria-label="SignalScope home">
      <span className="brand-mark__icon" aria-hidden="true"><span /><span /><span /></span>
      <span>SignalScope</span>
    </Link>
  );
}

export function SiteHeader() {
  const [open, setOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const [location] = useLocation();
  const links = [
    ["Analyze", "/analyze"],
    ["History", "/history"],
    ["Insights", "/insights"],
    ["Robustness", "/robustness"],
    ["How It Works", "/how-it-works"],
    ["About", "/about"],
  ];
  return (
    <header className="site-header">
      <div className="shell site-header__inner">
        <BrandMark />
        <nav className={`site-nav ${open ? "site-nav--open" : ""}`} aria-label="Primary navigation">
          {links.map(([label, href]) => (
            <Link key={href} href={href} onClick={() => setOpen(false)} className={location === href ? "is-active" : ""}>{label}</Link>
          ))}
        </nav>
        <div className="header-status"><span className="status-dot" /> Model online</div>
        <button className="theme-toggle" type="button" onClick={toggleTheme} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}><span className="theme-toggle__icon">{theme === "light" ? <Moon size={14} /> : <Sun size={14} />}</span><span className="theme-toggle__label">{theme === "light" ? "Dark" : "Light"}</span></button>
        <button className="icon-button menu-button" onClick={() => setOpen(!open)} aria-expanded={open} aria-label="Toggle menu"><Menu size={20} /></button>
      </div>
    </header>
  );
}

export function Footer() {
  return <footer className="site-footer"><div className="shell site-footer__inner"><div className="footer-brand"><BrandMark /><p>Telling real from synthetic in the age of generative media.</p></div><nav className="footer-links" aria-label="Footer navigation"><Link href="/analyze">Analyze</Link><Link href="/history">History</Link><Link href="/insights">Insights</Link><Link href="/robustness">Robustness</Link><Link href="/how-it-works">How It Works</Link><Link href="/about">About</Link></nav><div className="footer-meta"><span>Built for SIH 2026</span><span className="mono">SS / 2026</span></div></div></footer>;
}

export function SignalViewport() {
  const [signalStatus, setSignalStatus] = useState("READY");
  useEffect(() => {
    const statuses = ["READY", "SCANNING", "SIGNALS DETECTED", "READY"];
    let index = 0;
    const timer = window.setInterval(() => { index = (index + 1) % statuses.length; setSignalStatus(statuses[index]); }, 3600);
    return () => window.clearInterval(timer);
  }, []);
  return (
    <div className="signal-viewport" aria-hidden="true">
      <div className="viewport-label">SIGNAL MAP / 01</div>
      <div className="viewport-coordinates mono">X 048.120 &nbsp; Y 19.774</div>
      <div className="viewport-grid" />
      <div className="scan-frame"><i /><i /><i /><i /></div>
      <div className="signal-line signal-line--one" /><div className="signal-line signal-line--two" />
      <div className="signal-orbit" /><div className="signal-crosshair"><span /><span /></div>
      <div className="signal-marker signal-marker--one" /><div className="signal-marker signal-marker--two" /><div className="signal-wave" />
      <div className="signal-scanline" />
      <div className="viewport-footer mono"><span>VISUAL FEATURES</span><span className="signal-status"><i /> {signalStatus}</span></div>
    </div>
  );
}

const accepted = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export function UploadZone({ file, onFile, onRemove, disabled = false }: { file: File | null; onFile: (file: File) => void; onRemove: () => void; disabled?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const handleFile = (candidate?: File) => {
    if (!candidate) return;
    if (!accepted.includes(candidate.type)) { setError("Unsupported file. Use JPG, PNG, WEBP, or BMP."); return; }
    if (candidate.size > 15 * 1024 * 1024) { setError("This image is larger than 15 MB. Choose a smaller file."); return; }
    setError(""); onFile(candidate);
  };
  return (
    <div className="upload-wrap">
      <div className={`upload-zone ${dragging ? "is-dragging" : ""} ${file ? "has-file" : ""} ${disabled ? "is-disabled" : ""}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files?.[0]); }}>
        {!file ? <>
          <div className="upload-zone__topline"><span>INPUT / IMAGE</span><span className="mono">MAX 15 MB</span></div>
          <div className="upload-zone__center"><span className="upload-glyph"><Upload size={19} strokeWidth={1.6} /></span><div><strong>{dragging ? "Release to analyze" : "Drop an image to inspect"}</strong><p>{dragging ? "Your image will appear here for review" : <>or <button type="button" className="text-button" onClick={() => inputRef.current?.click()}>browse from your device</button></>}</p></div></div>
          <div className="upload-zone__meta"><span>JPG · PNG · WEBP · BMP</span><span className="mono">LOCAL UPLOAD</span></div>
        </> : <FilePreview file={file} onRemove={onRemove} />}
        <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" className="upload-input" onChange={(e) => handleFile(e.target.files?.[0])} />
      </div>
      {error && <p className="field-error" role="alert"><AlertTriangle size={15} />{error}</p>}
    </div>
  );
}

function FilePreview({ file, onRemove }: { file: File; onRemove: () => void }) {
  const [src] = useState(() => URL.createObjectURL(file));
  return <div className="file-preview"><img src={src} alt="Selected image preview" /><div className="file-preview__details"><span className="eyebrow">READY FOR ANALYSIS</span><strong>{file.name}</strong><span className="file-size mono">{(file.size / 1024 / 1024).toFixed(2)} MB <span>·</span> {file.type.split("/")[1]?.toUpperCase()}</span></div><button className="icon-button" onClick={onRemove} aria-label="Remove selected image"><X size={18} /></button></div>;
}

export function AnalysisLoader() {
  const stages = ["Loading image", "Extracting visual features", "Evaluating authenticity signals", "Generating visual explanation"];
  const [activeStage, setActiveStage] = useState(0);
  useEffect(() => { const timer = window.setInterval(() => setActiveStage((current) => Math.min(current + 1, stages.length - 1)), 900); return () => window.clearInterval(timer); }, [stages.length]);
  return <div className="analysis-loader"><div className="loader-top"><span className="eyebrow">LIVE ANALYSIS</span><span className="mono">PROCESSING</span></div><div className="loader-orb"><div className="loader-orb__ring" /><ScanLine size={23} /></div><h3>Analyzing visual signals<span className="ellipsis">...</span></h3><p>SignalScope is comparing this image against learned visual patterns.</p><div className="stage-list">{stages.map((stage, index) => <div className={`stage ${index <= activeStage ? "is-done" : ""}`} key={stage}><span className={`stage__marker ${index === activeStage ? "is-active" : ""}`}>{index < activeStage ? <Check size={14} /> : <LoaderCircle size={14} />}</span><span>{stage}</span><span className="stage__status">{index < activeStage ? "DONE" : index === activeStage ? "IN PROGRESS" : "QUEUED"}</span></div>)}</div></div>;
}

function percent(value = 0) { return `${Math.round(value * 100)}%`; }

const GENERATOR_NAMES: Record<string, string> = {
  dalle_3: "DALL-E 3",
  midjourney_6: "Midjourney 6",
  stable_diffusion_2_1: "Stable Diffusion 2.1",
  stable_diffusion_xl: "Stable Diffusion XL",
  stable_diffusion_3: "Stable Diffusion 3",
  real: "Real / Natural Image",
};

export function formatGeneratorName(rawName?: string): string {
  if (!rawName) return "Unknown";
  if (GENERATOR_NAMES[rawName]) {
    return GENERATOR_NAMES[rawName];
  }
  return rawName
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function GeneratorAttributionCard({ attribution }: { attribution?: GeneratorAttribution }) {
  if (!attribution || !attribution.generator) {
    return null;
  }

  let secondCandidate: GeneratorCandidate | null = null;
  if (Array.isArray(attribution.top_2) && attribution.top_2.length > 0) {
    const candidate = attribution.top_2.find(
      (item) => item && item.generator && item.generator !== attribution.generator
    ) || (attribution.top_2.length > 1 ? attribution.top_2[1] : null);

    if (candidate && candidate.generator && typeof candidate.confidence === "number") {
      secondCandidate = candidate;
    }
  }

  return (
    <div className="attribution-block">
      <div className="attribution-block__header">
        <span className="eyebrow">GENERATOR ATTRIBUTION</span>
      </div>
      <div className="attribution-card">
        <div className="attribution-main">
          <span className="attribution-label">Likely generator</span>
          <div className="attribution-primary-row">
            <h3 className="attribution-generator">{formatGeneratorName(attribution.generator)}</h3>
            <span className="attribution-confidence">{percent(attribution.confidence)}</span>
          </div>
        </div>
        {secondCandidate && (
          <div className="attribution-secondary">
            <span className="attribution-secondary__label">Also considered</span>
            <div className="attribution-secondary__row">
              <span className="attribution-secondary__name">{formatGeneratorName(secondCandidate.generator)}</span>
              <span className="attribution-secondary__conf mono">{percent(secondCandidate.confidence)}</span>
            </div>
          </div>
        )}
        <div className="attribution-disclaimer">
          <Info size={14} className="attribution-disclaimer__icon" />
          <p>Model-estimated attribution based on visual patterns. This is not provenance verification.</p>
        </div>
      </div>
    </div>
  );
}

export function ProvenanceCard({ provenance }: { provenance?: Provenance }) {
  if (!provenance) {
    return null;
  }

  const c2pa = provenance.c2pa || { status: "unavailable", verified: false };
  const exif = provenance.exif || { present: false, gps_present: false };

  let c2paStatusLabel = "C2PA Verification Unavailable";
  let c2paVariant: "verified" | "failed" | "normal" | "muted" = "muted";

  switch (c2pa.status) {
    case "verified":
      c2paStatusLabel = "Verified C2PA Content Credentials";
      c2paVariant = "verified";
      break;
    case "verification_failed":
      c2paStatusLabel = "C2PA Credentials Found — Verification Failed";
      c2paVariant = "failed";
      break;
    case "not_detected":
      c2paStatusLabel = "No C2PA Content Credentials Detected";
      c2paVariant = "normal";
      break;
    case "unavailable":
    default:
      c2paStatusLabel = "C2PA Verification Unavailable";
      c2paVariant = "muted";
      break;
  }

  const cameraInfo = [exif.camera_make, exif.camera_model].filter(Boolean).join(" ");
  const hasExifFields = Boolean(
    cameraInfo || exif.software || exif.datetime || exif.orientation || exif.gps_present
  );

  return (
    <div className="provenance-block">
      <div className="provenance-block__header">
        <span className="eyebrow">PROVENANCE & METADATA</span>
      </div>
      <div className="provenance-card">
        <div className="provenance-section">
          <div className="provenance-section__title">
            <ShieldCheck size={15} />
            <span>C2PA CONTENT CREDENTIALS</span>
          </div>
          <div className={`provenance-badge provenance-badge--${c2paVariant}`}>
            <span className="provenance-badge__dot" />
            <span>{c2paStatusLabel}</span>
          </div>
          {c2pa.verified && (c2pa.title || c2pa.creator) && (
            <dl className="provenance-details">
              {c2pa.title && (
                <div>
                  <dt>Title</dt>
                  <dd>{c2pa.title}</dd>
                </div>
              )}
              {c2pa.creator && (
                <div>
                  <dt>Creator</dt>
                  <dd>{c2pa.creator}</dd>
                </div>
              )}
            </dl>
          )}
        </div>

        <div className="provenance-section">
          <div className="provenance-section__title">
            <Camera size={15} />
            <span>EXIF METADATA</span>
          </div>
          {!exif.present ? (
            <p className="provenance-empty">No EXIF metadata detected</p>
          ) : (
            <dl className="provenance-details">
              {cameraInfo ? (
                <div>
                  <dt>Camera</dt>
                  <dd>{cameraInfo}</dd>
                </div>
              ) : null}
              {exif.software ? (
                <div>
                  <dt>Software</dt>
                  <dd>{exif.software}</dd>
                </div>
              ) : null}
              {exif.datetime ? (
                <div>
                  <dt>Date / Time</dt>
                  <dd>{exif.datetime}</dd>
                </div>
              ) : null}
              {exif.orientation ? (
                <div>
                  <dt>Orientation</dt>
                  <dd>{exif.orientation}</dd>
                </div>
              ) : null}
              {exif.gps_present ? (
                <div>
                  <dt>GPS Metadata</dt>
                  <dd className="provenance-gps">GPS metadata present</dd>
                </div>
              ) : null}
              {!hasExifFields && (
                <div>
                  <dt>Status</dt>
                  <dd>Metadata header present</dd>
                </div>
              )}
            </dl>
          )}
        </div>

        <div className="provenance-disclaimer">
          <Info size={14} className="provenance-disclaimer__icon" />
          <p>
            Provenance metadata provides additional context about an image's history when available. Its absence does not mean an image is AI-generated.
          </p>
        </div>
      </div>
    </div>
  );
}

export function ProbabilityBars({ result }: { result: AnalysisResult }) {
  return <div className="probability-block"><div className="probability-row"><div><span>AI Generated</span><strong>{percent(result.probability_ai)}</strong></div><div className="probability-track"><span className="probability-fill probability-fill--ai" style={{ "--fill": percent(result.probability_ai) } as React.CSSProperties} /></div></div><div className="probability-row"><div><span>Likely Real</span><strong>{percent(result.probability_real)}</strong></div><div className="probability-track"><span className="probability-fill probability-fill--real" style={{ "--fill": percent(result.probability_real) } as React.CSSProperties} /></div></div></div>;
}

export function ImageComparison({ file, heatmap }: { file: File; heatmap?: string }) {
  const [tab, setTab] = useState<"original" | "heatmap" | "overlay">("original");
  const [src] = useState(() => URL.createObjectURL(file));
  return <div className="comparison"><div className="comparison__header"><div><span className="eyebrow">VISUAL EVIDENCE</span><h3>Model attention</h3></div><div className="segmented-control" role="tablist" aria-label="Image view"><button className={tab === "original" ? "is-active" : ""} onClick={() => setTab("original")} role="tab">Original</button><button className={tab === "heatmap" ? "is-active" : ""} onClick={() => setTab("heatmap")} role="tab">Heatmap</button><button className={tab === "overlay" ? "is-active" : ""} onClick={() => setTab("overlay")} role="tab">Overlay</button></div></div><div className={`evidence-image evidence-image--${tab}`}><img src={tab === "original" ? src : (heatmap || src)} alt={tab === "original" ? "Original analyzed image" : "Grad-CAM model attention heatmap"} />{tab === "overlay" && heatmap && <img className="evidence-image__overlay" src={heatmap} alt="" aria-hidden="true" />}</div><div className="comparison__caption"><span><i className="caption-dot caption-dot--warm" /> Higher contribution</span><span className="mono">GRAD-CAM / EXPLANATION AID</span></div></div>;
}

export function ResultView({ result, file, onReset }: { result: AnalysisResult; file: File; onReset: () => void }) {
  const isAI = result.label.toLowerCase().includes("ai");
  return <div className="results-wrap">
    <div className="result-hero"><div className="result-hero__heading"><span className={`verdict-mark ${isAI ? "verdict-mark--ai" : "verdict-mark--real"}`}>{isAI ? <AlertTriangle size={22} /> : <Check size={22} />}</span><div><span className="eyebrow">MODEL ASSESSMENT</span><h2>Likely {isAI ? "AI Generated" : "Real"}</h2><p>{result.message || "The model has returned a confidence-based likelihood assessment for this image."}</p></div></div><div className="confidence"><span>CONFIDENCE</span><strong>{percent(result.confidence)}</strong><div className={`confidence-bar ${isAI ? "confidence-bar--ai" : "confidence-bar--real"}`}><i style={{ width: percent(result.confidence) }} /></div></div></div>
    <div className="result-grid"><div className="result-main"><ProbabilityBars result={result} />{result.attribution && <GeneratorAttributionCard attribution={result.attribution} />}<ImageComparison file={file} heatmap={result.heatmap_image} />{result.provenance && <ProvenanceCard provenance={result.provenance} />}<div className="signal-profile" aria-label="Decorative signal profile"><div className="signal-profile__head"><span className="eyebrow">VISUAL DESIGN ELEMENT</span><span className="mono">NOT MODEL DATA</span></div><div className="signal-profile__wave" /><p>Abstract signal profile for visual continuity — not a measurement returned by the classifier.</p></div></div><aside className="result-side"><div className="info-card"><div className="info-card__title"><Info size={16} /><span>Why this result?</span></div><p>Highlighted regions contributed more strongly to the model’s prediction. The Grad-CAM view is an explanation aid that helps inspect model attention, not proof of image origin.</p></div><div className="info-card info-card--muted"><div className="info-card__title"><ShieldCheck size={16} /><span>Use with care</span></div><p>Detection is probabilistic. Performance can vary across generators, transformations, compression, and image domains.</p></div><div className="tech-card"><div className="tech-card__title">TECHNICAL DETAILS</div><dl><div><dt>File name</dt><dd title={file.name}>{file.name}</dd></div><div><dt>File type</dt><dd>{file.type || "Unknown"}</dd></div><div><dt>Dimensions</dt><dd>Not returned</dd></div><div><dt>Model</dt><dd>SignalScope classifier</dd></div><div><dt>Status</dt><dd className="status-inline"><span className="status-dot" /> Complete</dd></div></dl></div></aside></div>
    <div className="result-actions"><button className="button button--primary" onClick={onReset}>Analyze another image <ArrowRight size={17} /></button><span className="result-note"><ShieldCheck size={15} /> No image is stored by this interface.</span></div>
  </div>;
}

export function PageIntro({ eyebrow, title, children }: { eyebrow: string; title: React.ReactNode; children?: React.ReactNode }) { return <div className="page-intro"><span className="eyebrow">{eyebrow}</span><h1>{title}</h1>{children && <p>{children}</p>}</div>; }

export function AccordionItem({ title, children }: { title: string; children: React.ReactNode }) { const [open, setOpen] = useState(false); return <div className={`accordion-item ${open ? "is-open" : ""}`}><button onClick={() => setOpen(!open)} aria-expanded={open}><span>{title}</span><ChevronDown size={18} /></button>{open && <div className="accordion-content">{children}</div>}</div>; }
