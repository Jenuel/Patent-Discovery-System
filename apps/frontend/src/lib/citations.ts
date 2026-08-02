import type { EvidenceItem } from '../types';

export interface Citation {
    /** `[3]` — the evidence ordinal, as the answer itself writes it. */
    label: string;
    ordinal: number;
    /** What the jump targets. Claim-level retrieval returns several chunks per
     *  patent, so the chunk is the only unique thing to point at. */
    chunkId: string;
    patentId: string;
}

/** Matches publication numbers the model may write out in full. */
const PATENT_NUMBER = /\b[A-Z]{2}\d{6,}(?:[A-Z]\d?)?\b/g;

/** Matches the bracketed ordinals the answer prompt asks the model to emit. */
const BRACKET_ORDINAL = /\[(\d{1,3})\]/g;

export const citationsFor = (answer: string, evidence: EvidenceItem[]): Citation[] => {
    const found = new Map<number, Citation>();

    const add = (ordinal: number) => {
        if (ordinal < 1 || ordinal > evidence.length || found.has(ordinal)) return;
        const item = evidence[ordinal - 1];
        found.set(ordinal, {
            label: `[${ordinal}]`,
            ordinal,
            chunkId: item.chunk_id,
            patentId: item.patent_id,
        });
    };

    for (const match of answer.matchAll(BRACKET_ORDINAL)) {
        add(Number.parseInt(match[1], 10));
    }

    if (found.size === 0) {
        const named = new Set(answer.match(PATENT_NUMBER) ?? []);
        evidence.forEach((item, index) => {
            if (named.has(item.patent_id)) add(index + 1);
        });
    }

    return [...found.values()].sort((a, b) => a.ordinal - b.ordinal);
};
