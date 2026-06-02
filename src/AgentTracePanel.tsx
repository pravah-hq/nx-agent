import { useMemo, useRef } from "react";
import {
  actionLabel,
  type AgentTraceStep,
  compactTraceId,
  parseAgentTrace,
} from "./agentTrace";

type Props = {
  steps: AgentTraceStep[];
  activeIndex: number;
  onStepsChange: (steps: AgentTraceStep[]) => void;
  onActiveIndexChange: (index: number) => void;
};

export function AgentTracePanel({
  steps,
  activeIndex,
  onStepsChange,
  onActiveIndexChange,
}: Props) {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const active = steps[activeIndex] ?? null;

  const loadText = (text: string) => {
    const parsed = parseAgentTrace(text);
    onStepsChange(parsed);
    onActiveIndexChange(0);
  };

  const summary = useMemo(() => {
    if (!steps.length) return null;
    const moves = steps.filter((s) => s.action === "move").length;
    const turns = steps.filter((s) => s.action === "turn_left" || s.action === "turn_right").length;
    const classifies = steps.filter((s) => s.action === "classify_or_stop").length;
    return `${steps.length} steps · ${moves} moves · ${turns} turns · ${classifies} classifies`;
  }, [steps]);

  return (
    <section className="trace-panel" aria-label="Agent trace">
      <header className="trace-header">
        <div>
          <p className="eyebrow">Agent trace</p>
          <strong>{summary ?? "Load a run log"}</strong>
        </div>
        <div className="trace-load-actions">
          <button type="button" onClick={() => fileRef.current?.click()}>
            Load JSON / log
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".json,.txt,.log,text/plain,application/json"
            hidden
            onChange={async (event) => {
              const file = event.target.files?.[0];
              if (!file) return;
              loadText(await file.text());
              event.target.value = "";
            }}
          />
          {steps.length > 0 && (
            <button type="button" className="ghost" onClick={() => onStepsChange([])}>
              Clear
            </button>
          )}
        </div>
      </header>

      {steps.length === 0 ? (
        <p className="trace-empty">
          Export from the agent with{" "}
          <code>python -m agent run --policy vlm --trace-out trace.json</code>, or upload verbose
          console output.
        </p>
      ) : (
        <>
          <div className="trace-scrubber">
            <label className="eyebrow" htmlFor="trace-step">
              Step {active?.step ?? 0}
            </label>
            <input
              id="trace-step"
              type="range"
              min={0}
              max={Math.max(0, steps.length - 1)}
              value={activeIndex}
              onChange={(event) => onActiveIndexChange(Number(event.target.value))}
            />
            <div className="trace-scrubber-meta">
              <span>
                {activeIndex + 1} / {steps.length}
              </span>
              {active && <span className="trace-action-badge">{actionLabel(active)}</span>}
            </div>
          </div>

          {active && (
            <div className="trace-step-detail">
              <div className="trace-grid">
                <div>
                  <p className="eyebrow">Position</p>
                  <span>
                    {compactTraceId(active.panoIdBefore)} → {compactTraceId(active.panoIdAfter)}
                  </span>
                </div>
                <div>
                  <p className="eyebrow">Facing bin</p>
                  <span>
                    {active.directionBinBefore}
                    {active.directionBinAfter !== active.directionBinBefore
                      ? ` → ${active.directionBinAfter}`
                      : ""}
                  </span>
                </div>
                <div>
                  <p className="eyebrow">Clear view</p>
                  <span>{active.poleInClearView ? "yes" : "no"}</span>
                </div>
                <div>
                  <p className="eyebrow">Visible pole</p>
                  <span>{active.visiblePoleId ?? "—"}</span>
                </div>
              </div>
              <p className="trace-message">{active.message || "—"}</p>

              <div className="trace-vlm-list">
                <p className="eyebrow">VLM output</p>
                {active.vlmCalls.length === 0 ? (
                  <span className="trace-vlm-empty">No VLM calls recorded for this step.</span>
                ) : (
                  active.vlmCalls.map((call, index) => (
                    <details key={`${call.phase}-${call.attempt}-${index}`} open={index === 0}>
                      <summary>
                        [{call.phase}
                        {call.attempt > 1 ? ` attempt ${call.attempt}` : ""}]
                      </summary>
                      <pre>{call.response}</pre>
                    </details>
                  ))
                )}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
