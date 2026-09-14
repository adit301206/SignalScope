import { useEffect, useState } from "react";
import { Activity, ArrowRight, BarChart3, Check, Clock3, Database, Gauge, History as HistoryIcon, Trash2, TriangleAlert } from "lucide-react";
import { Link } from "wouter";
import { clearHistory, deleteHistory, getHistory, type HistoryEntry } from "@/api/history";
import { PageIntro } from "@/components/SignalScopeUI";

export function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  useEffect(() => setEntries(getHistory()), []);
  return <div className="simple-page product-page"><div className="shell"><PageIntro eyebrow="LOCAL WORKSPACE" title={<>Analysis<br /><em>history.</em></>}><span>Your recent analyses stay on this device only. Nothing is uploaded here, and you can clear the record at any time.</span></PageIntro><div className="product-toolbar"><div><span className="eyebrow">RECENT ANALYSES</span><p>{entries.length ? `${entries.length} saved locally` : "No saved analyses"}</p></div>{entries.length > 0 && <button className="quiet-danger" onClick={() => { clearHistory(); setEntries([]); }}><Trash2 size={14} /> Clear history</button>}</div>{entries.length ? <div className="history-list">{entries.map((entry) => <HistoryRow key={entry.id} entry={entry} onDelete={() => { deleteHistory(entry.id); setEntries(getHistory()); }} />)}</div> : <div className="empty-product"><span className="empty-product__icon"><HistoryIcon size={22} /></span><h2>No analyses yet</h2><p>Your analyzed images will appear here.</p><Link href="/analyze" className="button button--primary">Analyze an image <ArrowRight size={16} /></Link></div>}</div></div>;
}

function HistoryRow({ entry, onDelete }: { entry: HistoryEntry; onDelete: () => void }) { const ai = entry.label.toLowerCase().includes("ai"); return <article className="history-row"><div className="history-thumb">{entry.thumbnail ? <img src={entry.thumbnail} alt="" /> : <Database size={20} />}</div><div className="history-info"><strong title={entry.filename}>{entry.filename || "Untitled image"}</strong><span className={`history-verdict ${ai ? "history-verdict--ai" : "history-verdict--real"}`}><i /> Likely {ai ? "AI Generated" : "Real"}</span><span className="mono">{Math.round(entry.confidence * 100)}% confidence · {new Date(entry.createdAt).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}</span></div><div className="history-actions"><Link href="/analyze" className="inline-link">Re-analyze <ArrowRight size={14} /></Link><button className="icon-button" onClick={onDelete} aria-label={`Delete ${entry.filename || "analysis"}`}><Trash2 size={16} /></button></div></article>; }

const unavailable = "Awaiting final evaluation";

export function InsightsPage() {
  const metrics = [
    ["ROC-AUC", "0.8790", "Held-out Defactify evaluation"],
    ["Macro-F1", "0.7303", "Held-out Defactify evaluation"],
    ["Accuracy", "82.00%", "Held-out Defactify evaluation"],
    ["False Positive Rate", "27.00%", "Real images classified as AI"],
    ["False Negative Rate", "16.20%", "AI images classified as real"],
  ];

  const generators = [
    ["DALL·E 3", "96.0%", "0.8780"],
    ["Midjourney 6", "85.0%", "0.7705"],
    ["SDXL", "83.0%", "0.7560"],
    ["Stable Diffusion 2.1", "82.0%", "0.7319"],
    ["Stable Diffusion 3", "73.0%", "0.6850"],
  ];

  return (
    <div className="simple-page product-page">
      <div className="shell">
        <PageIntro
          eyebrow="MODEL EVIDENCE"
          title={
            <>
              Does it
              <br />
              <em>generalize?</em>
            </>
          }
        >
          <span>
            SignalScope was evaluated on a separate Defactify distribution
            containing real images and images from five unseen generators.
          </span>
        </PageIntro>

        <div className="evidence-banner">
          <Check size={18} />
          <div>
            <strong>Held-out evaluation completed.</strong>
            <p>
              These measurements come from a separate 600-image Defactify
              evaluation set: 100 real images and 100 images from each of
              five generators.
            </p>
          </div>
        </div>

        <section className="metric-section">
          <div className="product-section-head">
            <div>
              <span className="eyebrow">OVERALL PERFORMANCE</span>
              <h2>Evaluation metrics</h2>
            </div>
            <span className="mono">HELD-OUT / UNSEEN GENERATORS</span>
          </div>

          <div className="metric-grid">
            {metrics.map(([label, value, note]) => (
              <div className="metric-card" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
                <small>{note}</small>
              </div>
            ))}
          </div>
        </section>

        <section className="evidence-layout">
          <div>
            <div className="product-section-head">
              <div>
                <span className="eyebrow">
                  UNSEEN GENERATOR EVALUATION
                </span>
                <h2>Beyond the training distribution</h2>
              </div>
            </div>

            <p className="section-copy">
              The detector was evaluated on images produced by generators
              outside the original CIFAKE training distribution. Detection
              performance varies across generators, highlighting why
              synthetic-image detection should be treated as a probabilistic
              assessment rather than proof of origin.
            </p>

            <div className="unavailable-table">
              {generators.map(([name, accuracy, probability]) => (
                <div key={name}>
                  <span>{name}</span>
                  <strong>{accuracy}</strong>
                  <small>Mean AI probability: {probability}</small>
                </div>
              ))}
            </div>
          </div>

          <div className="confusion-card">
            <div className="product-section-head">
              <div>
                <span className="eyebrow">CLASSIFICATION VIEW</span>
                <h2>Confusion matrix</h2>
              </div>
              <BarChart3 size={17} />
            </div>

            <div className="matrix">
              <span />
              <b>Real</b>
              <b>AI</b>

              <b>Real</b>
              <i>73</i>
              <i>27</i>

              <b>AI</b>
              <i>81</i>
              <i>419</i>
            </div>

            <p>
              On the 600-image held-out evaluation set, 73 real images were
              correctly identified as real and 419 AI images were correctly
              identified as AI.
            </p>
          </div>
        </section>

        <section className="metric-section">
          <div className="product-section-head">
            <div>
              <span className="eyebrow">GENERALIZATION SIGNAL</span>
              <h2>What changed?</h2>
            </div>
          </div>

          <div className="metric-grid">
            <div className="metric-card">
              <span>Baseline ROC-AUC</span>
              <strong>0.6174</strong>
              <small>Before mixed-data training</small>
            </div>

            <div className="metric-card">
              <span>Mixed-model ROC-AUC</span>
              <strong>0.8790</strong>
              <small>After mixed-data training</small>
            </div>

            <div className="metric-card">
              <span>AUC improvement</span>
              <strong>+0.2616</strong>
              <small>Absolute improvement</small>
            </div>

            <div className="metric-card">
              <span>Baseline FPR</span>
              <strong>100%</strong>
              <small>Baseline predicted all images as AI</small>
            </div>

            <div className="metric-card">
              <span>Mixed-model FPR</span>
              <strong>27%</strong>
              <small>Held-out Defactify evaluation</small>
            </div>
          </div>
        </section>

        <div className="page-cta">
          <Link
            href="/analyze"
            className="button button--primary"
          >
            Test an image <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  );
}

function EvidenceBanner() { return <div className="evidence-banner"><TriangleAlert size={18} /><div><strong>Evaluation data is configuration-ready.</strong><p>Connect verified measurements in the data configuration before presenting them in a demo. No placeholder statistics are shown as facts.</p></div></div>; }
function Metric({ label }: { label: string }) { return <div className="metric-card"><span>{label}</span><strong>{unavailable}</strong><small>Replace with measured result</small></div>; }

export function RobustnessPage() {
  const results = [
    {
      name: "Original",
      short: "Original",
      auc: "0.8788",
      f1: "0.7303",
      accuracy: "82.00%",
      fpr: "27.00%",
      fnr: "16.20%",
    },
    {
      name: "JPEG quality 50",
      short: "JPEG",
      auc: "0.8591",
      f1: "0.6710",
      accuracy: "75.00%",
      fpr: "22.00%",
      fnr: "25.60%",
    },
    {
      name: "Resize to 112",
      short: "Resize",
      auc: "0.7720",
      f1: "0.6052",
      accuracy: "84.33%",
      fpr: "80.00%",
      fnr: "2.80%",
    },
    {
      name: "Gaussian blur",
      short: "Blur",
      auc: "0.7973",
      f1: "0.6738",
      accuracy: "82.83%",
      fpr: "58.00%",
      fnr: "9.00%",
    },
    {
      name: "Brightness +15%",
      short: "Brightness",
      auc: "0.8820",
      f1: "0.7419",
      accuracy: "84.17%",
      fpr: "34.00%",
      fnr: "12.20%",
    },
  ];

  const transformations = [
    {
      number: "01",
      title: "JPEG compression",
      detail: "Quality 50",
      result: "AUC 0.8591",
      change: "−0.0197",
    },
    {
      number: "02",
      title: "Resizing",
      detail: "112 × 112 input",
      result: "AUC 0.7720",
      change: "−0.1068",
    },
    {
      number: "03",
      title: "Gaussian blur",
      detail: "Light blur",
      result: "AUC 0.7973",
      change: "−0.0815",
    },
    {
      number: "04",
      title: "Brightness",
      detail: "+15% brightness",
      result: "AUC 0.8820",
      change: "+0.0032",
    },
  ];

  return (
    <div className="simple-page product-page">
      <div className="shell">
        <PageIntro
          eyebrow="BONUS C / ROBUSTNESS"
          title={
            <>
              How stable is
              <br />
              <em>the detector?</em>
            </>
          }
        >
          <span>
            Real-world images are often compressed, resized, blurred, or lightly
            edited. Robustness testing measures how the detector behaves under
            these conditions.
          </span>
        </PageIntro>

        <div className="evidence-banner">
          <Check size={18} />
          <div>
            <strong>Held-out robustness evaluation completed.</strong>
            <p>
              Results are measured on the same 600-image Defactify evaluation
              set: 100 real images and 100 images from each of five generators.
            </p>
          </div>
        </div>

        <section className="metric-section">
          <div className="product-section-head">
            <div>
              <span className="eyebrow">DEGRADATION ANALYSIS</span>
              <h2>Performance under change</h2>
            </div>
            <span className="mono">600 HELD-OUT IMAGES</span>
          </div>

          <div className="robustness-grid">
            {transformations.map((item) => (
              <article className="robust-card" key={item.title}>
                <span className="robust-index mono">{item.number}</span>
                <Activity size={18} />

                <h3>{item.title}</h3>

                <span className="section-copy">{item.detail}</span>

                <div className="robust-flow">
                  <span>Original</span>
                  <ArrowRight size={14} />
                  <span>Transform</span>
                  <ArrowRight size={14} />
                  <span>Evaluate</span>
                </div>

                <strong>{item.result}</strong>
                <small>{item.change} vs original</small>
              </article>
            ))}
          </div>
        </section>

        <section className="metric-section">
          <div className="product-section-head">
            <div>
              <span className="eyebrow">MEASURED RESULTS</span>
              <h2>Robustness profile</h2>
            </div>
            <Gauge size={17} />
          </div>

          <div className="robustness-table">
            <div className="robustness-table__head">
              <span>Condition</span>
              <span>ROC-AUC</span>
              <span>Macro-F1</span>
              <span>Accuracy</span>
            </div>

            {results.map((item) => (
              <div
                className={`robustness-table__row ${
                  item.short === "Original"
                    ? "robustness-table__row--original"
                    : ""
                }`}
                key={item.name}
              >
                <strong>{item.name}</strong>
                <span>{item.auc}</span>
                <span>{item.f1}</span>
                <span>{item.accuracy}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="evidence-layout">
          <div>
            <div className="product-section-head">
              <div>
                <span className="eyebrow">AUC COMPARISON</span>
                <h2>Where does performance move?</h2>
              </div>
            </div>

            <div className="robustness-chart">
              {results.map((item) => {
                const width = Math.max(
                  10,
                  Math.round(parseFloat(item.auc) * 100)
                );

                return (
                  <div className="robustness-chart__row" key={item.name}>
                    <div className="robustness-chart__label">
                      <span>{item.short}</span>
                      <strong>{item.auc}</strong>
                    </div>

                    <div className="robustness-chart__track">
                      <div
                        className="robustness-chart__fill"
                        style={{ width: `${width}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="confusion-card">
            <div className="product-section-head">
              <div>
                <span className="eyebrow">KEY FINDING</span>
                <h2>What changed?</h2>
              </div>
              <TriangleAlert size={17} />
            </div>

            <p>
              Moderate JPEG compression caused a relatively small AUC drop,
              while resizing and blur produced larger degradation. Brightness
              adjustment remained essentially stable.
            </p>

            <div className="metric-grid">
              <div className="metric-card">
                <span>Best stability</span>
                <strong>Brightness</strong>
                <small>AUC 0.8820</small>
              </div>

              <div className="metric-card">
                <span>Largest AUC drop</span>
                <strong>Resize</strong>
                <small>−0.1068</small>
              </div>
            </div>
          </div>
        </section>

        <section className="metric-section">
          <div className="product-section-head">
            <div>
              <span className="eyebrow">ERROR PROFILE</span>
              <h2>False positives and negatives</h2>
            </div>
          </div>

          <div className="metric-grid">
            {results.slice(1).map((item) => (
              <div className="metric-card" key={item.name}>
                <span>{item.name}</span>
                <strong>{item.fpr} FPR</strong>
                <small>{item.fnr} FNR</small>
              </div>
            ))}
          </div>
        </section>

        <div className="page-cta">
          <Link
            href="/analyze"
            className="button button--primary"
          >
            Test an image <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  );
}
