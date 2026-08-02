import type { EvidenceItem, ResultFilters } from '../types';

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

/* ── facets ───────────────────────────────────────────────────────────── */

const LEVEL_ORDER = ['claim', 'limitation', 'patent'];
const SOURCE_ORDER = ['reranked', 'hybrid', 'dense', 'sparse'];

const orderedDistinct = (values: string[], order: string[]): string[] => {
    const seen = [...new Set(values.filter(Boolean).map((v) => v.toLowerCase()))];
    return seen.sort((a, b) => {
        const ai = order.indexOf(a);
        const bi = order.indexOf(b);
        if (ai === -1 && bi === -1) return a.localeCompare(b);
        if (ai === -1) return 1;
        if (bi === -1) return -1;
        return ai - bi;
    });
};

/** Chip values are read off the payload rather than hardcoded, so a response
 *  that only ever contains claims does not offer a dead `patent` chip. */
export const levelOptions = (evidence: EvidenceItem[]): string[] => [
    'all',
    ...orderedDistinct(evidence.map((e) => e.level), LEVEL_ORDER),
];

export const sourceOptions = (evidence: EvidenceItem[]): string[] => [
    'all',
    ...orderedDistinct(evidence.map((e) => e.source), SOURCE_ORDER),
];

export interface CpcFacet {
    /** `G06N3/*` for a single subgroup, `G06F*` when subgroups in a section split. */
    label: string;
    /** The literal prefix a code must start with to belong to this facet. */
    prefix: string;
    /** Documents carrying at least one code in this facet — not codes. */
    count: number;
}

/**
 * CPC codes seen in the result set, grouped one entry per subgroup (`G06N3/*`),
 * collapsed to the 4-character section (`G06F*`) when a section contributes more
 * than one. The corpus stores codes unpunctuated (`G06F1730259`), where there is
 * no subgroup boundary to read — those group at the section, so a label never
 * prints a whole code as if it were one.
 */
export const cpcFacets = (evidence: EvidenceItem[]): CpcFacet[] => {
    const sections = new Map<string, Set<string>>();

    for (const item of evidence) {
        for (const code of readCpcCodes(item)) {
            if (code.length < 4) continue;
            const section = code.slice(0, 4);
            // Only a punctuated code tells us where the subgroup ends.
            const subgroup = code.includes('/') ? code.split('/')[0] : section;
            if (!sections.has(section)) sections.set(section, new Set());
            sections.get(section)!.add(subgroup);
        }
    }

    const facets: CpcFacet[] = [];

    for (const [section, subgroups] of sections) {
        const [only] = [...subgroups];
        const collapse = subgroups.size > 1 || only === section;
        facets.push(
            collapse
                ? { label: `${section}*`, prefix: section, count: 0 }
                : { label: `${only}/*`, prefix: only, count: 0 },
        );
    }

    for (const facet of facets) {
        facet.count = evidence.filter((item) =>
            readCpcCodes(item).some((code) => code.startsWith(facet.prefix)),
        ).length;
    }

    return facets
        .filter((f) => f.count > 0)
        .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
};

/* ── narrowing ────────────────────────────────────────────────────────── */

export const applyFilters = (
    evidence: EvidenceItem[],
    filters: ResultFilters,
): EvidenceItem[] =>
    evidence.filter((item) => {
        if (filters.level !== 'all' && item.level?.toLowerCase() !== filters.level) return false;
        if (filters.source !== 'all' && item.source?.toLowerCase() !== filters.source) return false;
        // 1e-9 so a slider step landing exactly on a score keeps that score in.
        if (item.score < filters.minScore - 1e-9) return false;
        if (filters.cpcPrefix) {
            const codes = readCpcCodes(item);
            if (!codes.some((code) => code.startsWith(filters.cpcPrefix!))) return false;
        }
        return true;
    });
