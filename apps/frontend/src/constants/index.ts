import type { ComposerState, ResultFilters } from '../types';

export const APP_NAME = { head: 'Patent', tail: 'Discovery' } as const;

export const EMPTY_RESULT_FILTERS: ResultFilters = {
    level: 'all',
    source: 'all',
    minScore: 0,
    cpcPrefix: null,
};

/** Where a patent number resolves when the row's "Open patent" is followed. */
export const PATENT_URL = (patentId: string) =>
    `https://patents.google.com/patent/${encodeURIComponent(patentId)}`;

export const EMPTY_COMPOSER: ComposerState = {
    query: '',
    systemDescription: '',
    cpcCodes: '',
    yearFrom: '',
    yearTo: '',
};

export const HERO = {
    kicker: 'PRIOR ART · FREEDOM TO OPERATE · LANDSCAPES',
    title: "Know what's already patented.",
    lede:
        'Describe your invention in plain English. We read the claims of 142M filings ' +
        "and tell you which ones stand in your way — and which don't.",
} as const;

export const CAPABILITIES = [
    {
        title: 'A plain-English verdict',
        body: 'Clear, crowded, or blocked — with the reasoning shown.',
    },
    {
        title: 'Every claim it relies on',
        body: 'Each sentence links to the exact passage behind it.',
    },
    {
        title: 'A report for your counsel',
        body: 'Shortlist what matters, export as PDF or DOCX.',
    },
] as const;

export const COMPOSER_PLACEHOLDER =
    "Describe your invention — what it does, how it works, what's new about it…";

export const SEARCH_CTA = 'Search 142M filings';

export const EXAMPLE_QUERIES = [
    'on-device image classification without a server round trip',
    'solid-state battery cathode coating',
    'speaker diarization on embedded hardware',
] as const;

/* Carries the same corpus figure as HERO and SEARCH_CTA — retune together. */
export const STATS = [
    { value: '142M', label: 'filings across 34 offices' },
    { value: '40s', label: 'to a readable assessment' },
    { value: 'Claim-level', label: 'retrieval, not abstract keywords' },
    { value: 'Every cite', label: 'traceable to its source passage' },
] as const;

