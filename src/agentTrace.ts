export type VlmCall = {
  phase: string;
  attempt: number;
  response: string;
};

export type AgentTraceStep = {
  step: number;
  action: string;
  targetPanoId: string | null;
  poleType: string | null;
  poleInClearView: boolean;
  message: string;
  visiblePoleId: string | null;
  panoIdBefore: string;
  directionBinBefore: number;
  panoIdAfter: string;
  directionBinAfter: number;
  vlmCalls: VlmCall[];
};

export type AgentTraceDocument = {
  version: number;
  policy?: string;
  steps: AgentTraceStep[];
};

function normalizeVlmCall(raw: Record<string, unknown>): VlmCall {
  return {
    phase: String(raw.phase ?? "unknown"),
    attempt: Number(raw.attempt ?? 1),
    response: String(raw.response ?? ""),
  };
}

function normalizeStep(raw: Record<string, unknown>): AgentTraceStep {
  const before = (raw.state_before ?? raw.stateBefore ?? {}) as Record<string, unknown>;
  const after = (raw.state_after ?? raw.stateAfter ?? {}) as Record<string, unknown>;
  const vlmRaw = (raw.vlm_calls ?? raw.vlmCalls ?? []) as Record<string, unknown>[];

  return {
    step: Number(raw.step ?? 0),
    action: String(raw.action ?? "unknown"),
    targetPanoId: (raw.target_pano_id ?? raw.targetPanoId ?? null) as string | null,
    poleType: (raw.pole_type ?? raw.poleType ?? null) as string | null,
    poleInClearView: Boolean(raw.pole_in_clear_view ?? raw.poleInClearView ?? false),
    message: String(raw.message ?? ""),
    visiblePoleId: (raw.visible_pole_id ?? raw.visiblePoleId ?? null) as string | null,
    panoIdBefore: String(before.pano_id ?? before.panoId ?? ""),
    directionBinBefore: Number(before.direction_bin ?? before.directionBin ?? 0),
    panoIdAfter: String(after.pano_id ?? after.panoId ?? ""),
    directionBinAfter: Number(after.direction_bin ?? after.directionBin ?? 0),
    vlmCalls: vlmRaw.map(normalizeVlmCall),
  };
}

export function parseAgentTrace(text: string): AgentTraceStep[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (Array.isArray(parsed)) {
      return parsed.map((row) => normalizeStep(row as Record<string, unknown>));
    }
    if (parsed && typeof parsed === "object") {
      const doc = parsed as AgentTraceDocument & { steps?: unknown[] };
      if (Array.isArray(doc.steps)) {
        return doc.steps.map((row) => normalizeStep(row as Record<string, unknown>));
      }
    }
  } catch {
    /* fall through to text parser */
  }

  return parseTextAgentLog(trimmed);
}

function parseTextAgentLog(text: string): AgentTraceStep[] {
  const steps: AgentTraceStep[] = [];
  const chunks = text.split(/\n--- step (\d+) ---\n/);
  for (let i = 1; i < chunks.length; i += 2) {
    const stepNum = Number(chunks[i]);
    const body = chunks[i + 1] ?? "";
    const actionMatch = body.match(/action:\s*(\w+)\s*->\s*(.+)/);
    const action = actionMatch?.[1] ?? "unknown";
    const message = actionMatch?.[2]?.trim() ?? "";
    const clearMatch = body.match(/pole_in_clear_view:\s*(true|false)/i);
    const visibleMatch = body.match(/visible_pole_for_classify:\s*(\S+)/i);
    const panoMatch = body.match(/pano:\s*(\S+)/);

    const vlmCalls: VlmCall[] = [];
    const vlmSection = body.match(/vlm responses this step:\s*([\s\S]*?)(?=\naction:|$)/i);
    if (vlmSection) {
      const block = vlmSection[1];
      const phaseRegex = /\[(.+?)\]\s*\n([\s\S]*?)(?=\n\s*\[|$)/g;
      let match: RegExpExecArray | null;
      while ((match = phaseRegex.exec(block)) !== null) {
        const label = match[1];
        const attemptMatch = label.match(/^(.+?) \(attempt (\d+)\)$/);
        vlmCalls.push({
          phase: attemptMatch?.[1] ?? label,
          attempt: attemptMatch ? Number(attemptMatch[2]) : 1,
          response: match[2].trim(),
        });
      }
    }

    steps.push({
      step: stepNum,
      action,
      targetPanoId: null,
      poleType: action === "classify_or_stop" ? null : null,
      poleInClearView: clearMatch?.[1]?.toLowerCase() === "true",
      message,
      visiblePoleId: visibleMatch?.[1] === "none" ? null : visibleMatch?.[1] ?? null,
      panoIdBefore: panoMatch?.[1] ?? "",
      directionBinBefore: 0,
      panoIdAfter: panoMatch?.[1] ?? "",
      directionBinAfter: 0,
      vlmCalls,
    });
  }
  return steps;
}

export const DIRECTION_BIN_WIDTH_DEG = 30;

export function viewYawDeg(panoHeadingDeg: number, directionBin: number): number {
  return ((panoHeadingDeg + directionBin * DIRECTION_BIN_WIDTH_DEG) % 360 + 360) % 360;
}

export function actionLabel(step: AgentTraceStep): string {
  if (step.action === "move" && step.targetPanoId) {
    return `move → ${compactTraceId(step.targetPanoId)}`;
  }
  if (step.action === "classify_or_stop" && step.poleType) {
    return `classify ${step.poleType}`;
  }
  return step.action.replace(/_/g, " ");
}

export function compactTraceId(value: string): string {
  const parts = value.split("/");
  return parts.at(-1) ?? value;
}
