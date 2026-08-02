import type { EvidenceItem } from '../types';

/* ── field readers ─────────────────────────────────────────────────────────
   `metadata` is whatever the chunk document carried, so every read below is a
   lookup across the spellings ingestion is known to produce plus the obvious
   aliases. A field that is genuinely absent returns null, which lets the
   caller drop that segment rather than print a placeholder. */

const firstString = (
    metadata: Record<string, unknown>,
    keys: string[],
): string | null => {
    for (const key of keys) {
        const value = metadata[key];
        if (typeof value === 'string' && value.trim()) return value.trim();
        if (typeof value === 'number') return String(value);
    }
    return null;
};

/** HUPD chunks carry no assignee; other ingestions may. Absent → omit it. */
export const readAssignee = (item: EvidenceItem): string | null =>
    firstString(item.metadata, ['assignee', 'assignee_name', 'applicant', 'applicant_name']);

export const readYear = (item: EvidenceItem): string | null =>
    firstString(item.metadata, ['year', 'publication_year', 'filing_year', 'priority_year']);

const asStringList = (raw: unknown): string[] => {
    if (Array.isArray(raw)) return raw.filter((v): v is string => typeof v === 'string' && !!v);
    return typeof raw === 'string' && raw ? [raw] : [];
};

/** Ingestion spells this `cpc_codes` in one place and `cpc` in another, and
 *  `cpc_prefix` is sometimes a string and sometimes a list — read all of them. */
export const readCpcCodes = (item: EvidenceItem): string[] => {
    const codes = asStringList(item.metadata.cpc_codes ?? item.metadata.cpc);
    if (codes.length) return codes;
    return asStringList(item.metadata.cpc_prefix);
};

/* ── row labels ───────────────────────────────────────────────────────── */

export const levelLabel = (item: EvidenceItem): string => {
    if (item.level?.toLowerCase() === 'claim') {
        return item.claim_no != null ? `CLAIM ${item.claim_no}` : 'CLAIM';
    }
    return `${(item.level || 'patent').toUpperCase()} LEVEL`;
};

export const isClaimLevel = (item: EvidenceItem): boolean =>
    item.level?.toLowerCase() === 'claim';

/** The reranked arm is the one the design tints — the scored-by-model pass. */
export const isReranked = (item: EvidenceItem): boolean =>
    item.source?.toLowerCase() === 'reranked';

export const modeLabel = (mode: string): string => mode.replace(/_/g, ' ').toUpperCase();

/** Four decimals — retrieval scores separate late. */
export const formatScore = (score: number): string => score.toFixed(4);

export const barWidth = (score: number): string =>
    `${Math.max(0, Math.min(100, Math.round(score * 100)))}%`;
