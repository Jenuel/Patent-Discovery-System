
export type QueryMode = 'prior_art' | 'infringement' | 'landscape';

export type EvidenceLevel = 'patent' | 'claim' | 'limitation';

export type EvidenceSource = 'dense' | 'sparse' | 'hybrid' | 'reranked';

/** Free-form on the wire — `metadata` is whatever the chunk document carried. */
export interface EvidenceMetadata {
    year?: number | string;
    cpc_codes?: string[];
    cpc_prefix?: string;
    assignee?: string;
    [key: string]: unknown;
}

export interface EvidenceItem {
    chunk_id: string;
    patent_id: string;
    /** patent | claim | limitation */
    level: string;
    title?: string | null;
    claim_no?: number | null;
    text: string;
    score: number;
    /** dense | sparse | hybrid | reranked */
    source: string;
    metadata: EvidenceMetadata;
}

export interface QueryResponse {
    mode: string;
    answer: string;
    evidence: EvidenceItem[];
}

export interface QueryFilters {
    cpc_prefixes?: string[];
    year_from?: number;
    year_to?: number;
}

export interface QueryRequest {
    query: string;
    top_k?: number;
    system_description?: string;
    filters?: QueryFilters;
}

/** Client-side narrowing of an already-returned evidence list. */
export interface ResultFilters {
    level: string;
    source: string;
    minScore: number;
    cpcPrefix: string | null;
}

/** What the composer collects before it is shaped into a QueryRequest. */
export interface ComposerState {
    query: string;
    systemDescription: string;
    cpcCodes: string;
    yearFrom: string;
    yearTo: string;
}
