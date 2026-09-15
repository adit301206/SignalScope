import { useEffect, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  Server,
  UploadCloud,
} from "lucide-react";
import { Link } from "wouter";

import {
  analyzeImage,
  type AnalysisResult,
} from "@/api/analysis";

import {
  createHistoryThumbnail,
  getHistory,
  saveHistory,
} from "@/api/history";

import {
  AnalysisLoader,
  ResultView,
  UploadZone,
} from "@/components/SignalScopeUI";

type State =
  | "empty"
  | "ready"
  | "loading"
  | "result"
  | "error";

/**
 * Converts a locally stored data URL back into a File object.
 *
 * Used when the user clicks "Re-analyze" from History.
 */
function dataUrlToFile(
  dataUrl: string,
  filename: string,
  fileType = "image/jpeg"
): File {
  const [header, base64] =
    dataUrl.split(",");

  if (!base64) {
    throw new Error(
      "invalid-image-data"
    );
  }

  const mimeMatch =
    header.match(
      /data:(.*?);base64/
    );

  const mime =
    mimeMatch?.[1] ||
    fileType;

  const binary =
    atob(base64);

  const bytes =
    new Uint8Array(
      binary.length
    );

  for (
    let i = 0;
    i < binary.length;
    i++
  ) {
    bytes[i] =
      binary.charCodeAt(i);
  }

  return new File(
    [bytes],
    filename,
    {
      type: mime,
    }
  );
}

export default function Analyze() {
  const [file, setFile] =
    useState<File | null>(null);

  const [result, setResult] =
    useState<AnalysisResult | null>(
      null
    );

  const [state, setState] =
    useState<State>("empty");

  const [error, setError] =
    useState("");

  /*
   * Restore an image when coming from:
   *
   * /analyze?historyId=<id>
   *
   * The History page passes the ID
   * of the saved entry.
   */
  useEffect(() => {
    const params =
      new URLSearchParams(
        window.location.search
      );

    const historyId =
      params.get("historyId");

    if (!historyId) {
      return;
    }

    const entry =
      getHistory().find(
        (item) =>
          item.id === historyId
      );

    if (entry?.thumbnail) {
      try {
        const restoredFile =
          dataUrlToFile(
            entry.thumbnail,
            entry.filename ||
              "re-analyzed-image.jpg",
            entry.fileType ||
              "image/jpeg"
          );

        setFile(
          restoredFile
        );

        setResult(null);
        setError("");
        setState("ready");

        return;
      } catch {
        // Fall through to clearing parameter.
      }
    }

    /*
     * If the history entry has no thumbnail,
     * cleanly remove the historyId parameter.
     */
    window.history.replaceState(
      {},
      "",
      "/analyze"
    );
  }, []);

  const chooseFile = (
    next: File
  ) => {
    setFile(next);
    setResult(null);
    setError("");
    setState("ready");
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError("");
    setState("empty");

    // Remove historyId from the URL.
    window.history.replaceState(
      {},
      "",
      "/analyze"
    );
  };

  const runAnalysis =
    async () => {
      if (!file) {
        return;
      }

      setState("loading");
      setError("");

      try {
        const response =
          await analyzeImage(file);

        setResult(response);
        setState("result");

        /*
         * Save lightweight history metadata
         * plus a SMALL compressed thumbnail.
         *
         * The full image and Grad-CAM heatmap
         * are NOT stored in localStorage.
         */
        try {
          const thumbnail =
            await createHistoryThumbnail(
              file
            );

          saveHistory(
            response,
            file.type,
            thumbnail
          );
        } catch (saveErr) {
          /*
           * Analysis should still succeed even
           * if thumbnail generation/storage fails.
           */
          console.warn(
            "Could not save thumbnail to local history:",
            saveErr
          );

          saveHistory(
            response,
            file.type
          );
        }
      } catch (err) {
        let message =
          "The image could not be analyzed right now. Try again with another image.";

        if (
          err instanceof Error
        ) {
          if (
            err.message ===
            "backend-unavailable"
          ) {
            message =
              "We couldn't reach the analysis service. Check that the SignalScope backend is running and try again.";
          } else if (
            err.message ===
            "file-too-large"
          ) {
            message =
              "The uploaded image file is too large. Please select an image under 15 MB.";
          } else if (
            err.message ===
            "invalid-response"
          ) {
            message =
              "Received an invalid response format from the analysis service.";
          } else if (
            err.message ===
            "invalid-image-data"
          ) {
            message =
              "The saved image data is invalid. Please upload the image again.";
          } else if (
            err.message
          ) {
            message =
              err.message;
          }
        }

        setError(message);
        setState("error");
      }
    };

  return (
    <div className="analyze-page">
      <div className="shell">
        <div className="analyze-header">
          <div>
            <span className="eyebrow eyebrow--rule">
              ANALYSIS WORKSPACE{" "}
              <span>LIVE</span>
            </span>

            <h1>
              Analyze an image
            </h1>

            <p>
              Inspect the likelihood of
              synthetic generation with
              visual evidence from the
              model.
            </p>
          </div>

          <div className="analysis-state">
            <span
              className={`status-dot ${
                state === "loading"
                  ? "status-dot--pulse"
                  : ""
              }`}
            />

            {state === "loading"
              ? "Analyzing"
              : "Model online"}
          </div>
        </div>

        {state === "result" &&
        file &&
        result ? (
          <ResultView
            result={result}
            file={file}
            onReset={reset}
          />
        ) : (
          <div className="analysis-workspace">
            <div className="analysis-workspace__main">
              <div className="workspace-label">
                <span>01</span>
                <span>
                  SELECT SOURCE
                </span>
              </div>

              <UploadZone
                file={file}
                onFile={chooseFile}
                onRemove={reset}
                disabled={
                  state === "loading"
                }
              />

              {state ===
              "loading" ? (
                <AnalysisLoader />
              ) : (
                <div className="analysis-cta">
                  <div className="analysis-cta__hint">
                    {state ===
                    "empty" ? (
                      <>
                        <UploadCloud
                          size={17}
                        />
                        Start with one
                        image
                      </>
                    ) : state ===
                      "error" ? (
                      <>
                        <AlertCircle
                          size={17}
                        />
                        Analysis stopped.
                        Try again.
                      </>
                    ) : (
                      <>
                        <span className="status-dot" />
                        Image ready to
                        inspect
                      </>
                    )}
                  </div>

                  <button
                    className="button button--primary"
                    disabled={
                      !file ||
                      state === "empty"
                    }
                    onClick={
                      runAnalysis
                    }
                  >
                    {state === "error"
                      ? "Try again"
                      : "Analyze image"}

                    <ArrowRight
                      size={17}
                    />
                  </button>
                </div>
              )}
            </div>

            <aside className="analysis-workspace__aside">
              <div className="aside-panel">
                <div className="aside-panel__header">
                  <span className="eyebrow">
                    BEFORE YOU BEGIN
                  </span>

                  <span className="mono">
                    02
                  </span>
                </div>

                <h2>
                  One image.
                  <br />
                  <em>
                    A closer look.
                  </em>
                </h2>

                <p>
                  SignalScope works best
                  when the original image
                  is clear and has not been
                  heavily recompressed.
                </p>

                <ul>
                  <li>
                    Accepted: JPG, PNG,
                    WEBP, BMP
                  </li>

                  <li>
                    Maximum file size:
                    15 MB
                  </li>

                  <li>
                    Images are not stored
                    on the analysis server.
                    If you use History, a
                    local thumbnail is kept
                    in your browser.
                  </li>
                </ul>
              </div>

              <div className="aside-panel aside-panel--technical">
                <div className="aside-panel__header">
                  <span className="eyebrow">
                    MODEL STATUS
                  </span>

                  <Server
                    size={15}
                  />
                </div>

                <div className="model-status-row">
                  <span>
                    SignalScope classifier
                  </span>

                  <span className="status-badge">
                    <i /> Ready
                  </span>
                </div>

                <div className="model-status-row">
                  <span>
                    Explanation layer
                  </span>

                  <span className="status-badge">
                    <i /> Ready
                  </span>
                </div>
              </div>
            </aside>
          </div>
        )}

        {error &&
        state !== "error"
          ? null
          : error && (
              <div
                className="service-error"
                role="alert"
              >
                <AlertCircle
                  size={17}
                />

                <div>
                  <strong>
                    Analysis unavailable
                  </strong>

                  <p>
                    {error}
                  </p>
                </div>
              </div>
            )}

        <div className="workspace-footnote">
          <span className="mono">
            03 / NOTE
          </span>

          <p>
            SignalScope provides a
            likelihood assessment, not
            definitive proof of image
            origin. Performance can vary
            across image domains and
            transformations.
          </p>

          <Link href="/about">
            Learn about limitations
            <ArrowRight size={14} />
          </Link>
        </div>
      </div>
    </div>
  );
}