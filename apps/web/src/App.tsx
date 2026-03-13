/** Main UI flow for PDF upload, streamed pipeline progress, and report download. */

import { useCallback, useMemo, useRef, useState } from "react";
import { QueryClient, QueryClientProvider, useMutation } from "@tanstack/react-query";
import { Download, FileText, FileUp } from "lucide-react";
import { uploadRun, streamPipeline, reportUrl, type PipelineEvent } from "./api";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { pdfFileSchema } from "./validation";

const qc = new QueryClient();

type StepLog = { at: string; msg: string; isError?: boolean };

type SummaryStats = {
  green: number;
  orange: number;
  blue: number;
  nuggets: number;
  critical: number;
  total_a?: number;
  total_b?: number;
};

const STATUS_COLORS: Record<string, string> = {
  green: "bg-green-100 text-green-900",
  orange: "bg-orange-100 text-orange-900",
  blue: "bg-blue-100 text-blue-900",
  nugget: "bg-yellow-100 text-yellow-900",
  critical: "bg-red-100 text-red-900",
};

const STEP_LABELS: Record<string, string> = {
  extract: "Extracting line items",
  map_rooms: "Mapping rooms",
  match: "Matching items",
  render: "Rendering PDF",
  done: "Complete",
};

function money(x: number | undefined) {
  if (x == null) return "—";
  return `$${x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function AppInner() {
  const [a, setA] = useState<File | null>(null);
  const [b, setB] = useState<File | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [activeStep, setActiveStep] = useState<string | null>(null);
  const [logs, setLogs] = useState<StepLog[]>([]);
  const [summary, setSummary] = useState<SummaryStats | null>(null);
  const [renderReady, setRenderReady] = useState(false);
  const cleanupRef = useRef<(() => void) | null>(null);

  const uploadMutation = useMutation({
    mutationFn: ({ proposalA, proposalB }: { proposalA: File; proposalB: File }) =>
      uploadRun(proposalA, proposalB),
  });

  const addLog = useCallback(
    (msg: string, isError = false) =>
      setLogs((l) => [...l, { at: new Date().toLocaleTimeString(), msg, isError }]),
    [],
  );

  const validatePdfOrThrow = (file: File | null, label: string): File => {
    const parsed = pdfFileSchema.safeParse(file);
    if (!parsed.success) throw new Error(`${label}: ${parsed.error.issues[0]?.message ?? "Invalid file."}`);
    return parsed.data;
  };

  const onSelectA = (file: File | null) => {
    if (!file) { setA(null); return; }
    const parsed = pdfFileSchema.safeParse(file);
    if (!parsed.success) { addLog(`Proposal A: ${parsed.error.issues[0]?.message ?? "Invalid file."}`, true); setA(null); return; }
    setA(parsed.data);
  };

  const onSelectB = (file: File | null) => {
    if (!file) { setB(null); return; }
    const parsed = pdfFileSchema.safeParse(file);
    if (!parsed.success) { addLog(`Proposal B: ${parsed.error.issues[0]?.message ?? "Invalid file."}`, true); setB(null); return; }
    setB(parsed.data);
  };

  const handleEvent = useCallback(
    (event: PipelineEvent) => {
      const label = STEP_LABELS[event.step] ?? event.step;
      if (event.status === "start") {
        setActiveStep(event.step);
        addLog(`[${label}] ${event.msg ?? "Starting…"}`);
      } else if (event.status === "done") {
        setActiveStep(event.step === "done" ? null : event.step);
        addLog(`[${label}] ${event.msg ?? "Done."}`);

        // Extract summary stats from match event
        if (event.step === "match" && event.data) {
          const d = event.data as Record<string, unknown>;
          const sc = (d.status_counts ?? {}) as Record<string, number>;
          const audit = (d.coverage_audit ?? {}) as Record<string, unknown>;
          setSummary({
            green: sc.green ?? 0,
            orange: sc.orange ?? 0,
            blue: sc.blue ?? 0,
            nuggets: (d.nugget_count as number) ?? 0,
            critical: (audit.critical_blue_count as number) ?? 0,
          });
        }

        if (event.step === "render") {
          setRenderReady(true);
        }
        if (event.step === "done") {
          setBusy(false);
          setActiveStep(null);
        }
      } else if (event.status === "error") {
        addLog(`ERROR [${label}]: ${event.msg ?? "Unknown error."}`, true);
        setBusy(false);
        setActiveStep(null);
      }
    },
    [addLog],
  );

  const runPipeline = async () => {
    if (!a || !b) return;
    try {
      const proposalA = validatePdfOrThrow(a, "Proposal A");
      const proposalB = validatePdfOrThrow(b, "Proposal B");

      setBusy(true);
      setLogs([]);
      setRunId(null);
      setSummary(null);
      setActiveStep(null);
      setRenderReady(false);

      addLog("Uploading PDFs…");
      const up = await uploadMutation.mutateAsync({ proposalA, proposalB });
      setRunId(up.run_id);
      addLog(`Uploaded. run_id=${up.run_id}`);

      addLog("Starting pipeline stream…");
      const cleanup = streamPipeline(up.run_id, handleEvent, (err) => {
        addLog(`Stream error: ${err.message}`, true);
        setBusy(false);
        setActiveStep(null);
      });
      cleanupRef.current = cleanup;
    } catch (e) {
      addLog(`ERROR: ${e instanceof Error ? e.message : String(e)}`, true);
      setBusy(false);
      setActiveStep(null);
    }
  };

  const downloadHref = useMemo(() => (runId && renderReady ? reportUrl(runId) : null), [runId, renderReady]);
  const isDone = !busy && runId != null;

  return (
    <div className="min-h-screen bg-[radial-gradient(70rem_24rem_at_50%_-10%,#ffffff_0%,transparent_60%),linear-gradient(135deg,#091826_0%,#102539_42%,#dfe8ef_100%)] p-4 sm:p-6">
      <Card className="mx-auto w-full max-w-5xl border border-stone-300/70 bg-[#fffdf7]/95 shadow-[0_24px_70px_rgba(7,17,28,0.30)] backdrop-blur-sm">
        <CardHeader className="space-y-5 border-b border-stone-300 pb-5">
          <div className="flex items-start gap-3">
            <div
              aria-hidden="true"
              className="mt-1 h-11 w-11 rounded-[4px] border border-white/60 bg-[linear-gradient(180deg,#ad2f2a_0_55%,#1a6c71_55%_100%)] shadow-[0_0_0_1px_rgba(255,255,255,0.4)_inset]"
            />
            <div>
              <p className="text-[0.66rem] font-semibold uppercase tracking-[0.12em] text-teal-700">
                Ciridae Estimate Review
              </p>
              <CardTitle className="font-serif text-3xl leading-tight text-stone-900 sm:text-4xl">
                Restoration Proposal Comparator
              </CardTitle>
              <CardDescription className="mt-1 max-w-2xl text-sm text-stone-600">
                Upload two PDF estimates and generate a matched reconciliation report.
              </CardDescription>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="border-t border-stone-300 pt-2">
              <p className="text-[0.7rem] uppercase tracking-[0.09em] text-stone-500">Claim ID</p>
              <p className="text-sm font-semibold text-stone-900">
                {runId ? runId.slice(0, 8).toUpperCase() : "PENDING"}
              </p>
            </div>
            <div className="border-t border-stone-300 pt-2">
              <p className="text-[0.7rem] uppercase tracking-[0.09em] text-stone-500">Active Step</p>
              <p className="text-sm font-semibold text-stone-900">
                {activeStep ? (STEP_LABELS[activeStep] ?? activeStep) : isDone ? "Complete" : "—"}
              </p>
            </div>
            <div className="border-t border-stone-300 pt-2">
              <p className="text-[0.7rem] uppercase tracking-[0.09em] text-stone-500">Status</p>
              <Badge className={busy ? "mt-1 bg-amber-100 text-amber-900" : isDone ? "mt-1 bg-teal-100 text-teal-900" : "mt-1 bg-stone-100 text-stone-700"}>
                {busy ? "Processing" : isDone ? "Done" : "Ready"}
              </Badge>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4 p-4 sm:p-6">
          {/* Upload row */}
          <section className="grid gap-3 lg:grid-cols-2">
            <Card className="border border-stone-300 bg-[#ffffffcc] shadow-none">
              <CardHeader className="pb-3">
                <CardTitle className="font-serif text-lg">Contractor Proposal (A)</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <label className="block rounded-md border border-dashed border-stone-400 bg-stone-50 p-3">
                  <p className="mb-2 text-sm text-stone-700">{a?.name ?? "Select contractor estimate PDF"}</p>
                  <Input
                    type="file"
                    accept="application/pdf"
                    onChange={(e) => onSelectA(e.target.files?.[0] ?? null)}
                    disabled={busy}
                    className="h-9 border-stone-300 bg-white file:mr-3 file:rounded-md file:border-0 file:bg-stone-900 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-stone-100 hover:file:bg-stone-700"
                  />
                </label>
              </CardContent>
            </Card>

            <Card className="border border-stone-300 bg-[#ffffffcc] shadow-none">
              <CardHeader className="pb-3">
                <CardTitle className="font-serif text-lg">Insurance Proposal (B)</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <label className="block rounded-md border border-dashed border-stone-400 bg-stone-50 p-3">
                  <p className="mb-2 text-sm text-stone-700">{b?.name ?? "Select insurance estimate PDF"}</p>
                  <Input
                    type="file"
                    accept="application/pdf"
                    onChange={(e) => onSelectB(e.target.files?.[0] ?? null)}
                    disabled={busy}
                    className="h-9 border-stone-300 bg-white file:mr-3 file:rounded-md file:border-0 file:bg-stone-900 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-stone-100 hover:file:bg-stone-700"
                  />
                </label>
              </CardContent>
            </Card>
          </section>

          {/* Actions */}
          <section className="flex flex-wrap gap-3">
            <Button
              onClick={runPipeline}
              disabled={busy || !a || !b}
              className="bg-gradient-to-r from-teal-700 to-sky-700 text-white hover:from-teal-600 hover:to-sky-600"
            >
              <FileUp className="mr-2 h-4 w-4" />
              {busy ? "Running…" : "Run Full Comparison"}
            </Button>
            {downloadHref && !busy && (
              <Button asChild variant="outline" className="border-stone-400 bg-white/70 hover:bg-stone-100">
                <a href={downloadHref} target="_blank" rel="noreferrer">
                  <Download className="mr-2 h-4 w-4" />
                  Download PDF Report
                </a>
              </Button>
            )}
          </section>

          {/* Summary stats panel (shown after matching completes) */}
          {summary && (
            <section>
              <Card className="border border-stone-300 bg-white/80 shadow-none">
                <CardHeader className="border-b border-stone-300 pb-3">
                  <CardTitle className="font-serif text-xl">Match Summary</CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
                    {[
                      { label: "Green", key: "green", color: STATUS_COLORS.green, value: summary.green },
                      { label: "Orange", key: "orange", color: STATUS_COLORS.orange, value: summary.orange },
                      { label: "Blue (JDR-only)", key: "blue", color: STATUS_COLORS.blue, value: summary.blue },
                      { label: "Nuggets", key: "nugget", color: STATUS_COLORS.nugget, value: summary.nuggets },
                      { label: "Critical", key: "critical", color: STATUS_COLORS.critical, value: summary.critical },
                    ].map(({ label, key, color, value }) => (
                      <div key={key} className="rounded-md border border-stone-200 p-3 text-center">
                        <p className="text-xs uppercase tracking-wide text-stone-500">{label}</p>
                        <span className={`mt-1 inline-block rounded-full px-3 py-1 text-lg font-bold ${color}`}>
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>
                  {(summary.total_a != null || summary.total_b != null) && (
                    <div className="mt-3 flex flex-wrap gap-4 border-t border-stone-200 pt-3 text-sm">
                      <span className="text-stone-600">
                        JDR total: <strong>{money(summary.total_a)}</strong>
                      </span>
                      <span className="text-stone-600">
                        Insurance total: <strong>{money(summary.total_b)}</strong>
                      </span>
                      {summary.total_a != null && summary.total_b != null && (
                        <span className="text-stone-600">
                          Gap: <strong>{money(Math.abs(summary.total_a - summary.total_b))}</strong>
                        </span>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </section>
          )}

          {/* Processing log */}
          <section>
            <Card className="border border-stone-300 bg-white/70 shadow-none">
              <CardHeader className="border-b border-stone-300 pb-3">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-stone-600" />
                  <CardTitle className="font-serif text-xl">Processing Log</CardTitle>
                </div>
                <CardDescription>Real-time stream: extract → room map → match → render.</CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                {logs.length === 0 ? (
                  <p className="p-4 text-sm text-stone-600">No activity yet.</p>
                ) : (
                  <div className="divide-y divide-stone-200">
                    {logs.map((l, i) => (
                      <div
                        className="grid gap-2 px-4 py-3 text-sm opacity-0 [animation:fadeIn_260ms_ease_forwards] sm:grid-cols-[6rem,1fr]"
                        key={`${l.at}-${i}`}
                        style={{ animationDelay: `${Math.min(i * 45, 300)}ms` }}
                      >
                        <time className="font-mono text-xs text-stone-500">{l.at}</time>
                        <p className={`leading-relaxed ${l.isError ? "text-red-700 font-medium" : "text-stone-800"}`}>
                          {l.msg}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </section>
        </CardContent>
      </Card>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <AppInner />
    </QueryClientProvider>
  );
}
