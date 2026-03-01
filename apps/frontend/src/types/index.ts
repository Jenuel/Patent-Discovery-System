export interface EvidenceChunk {
    patentId: string;
    title: string;
    score: number;
    sourceType: 'dense' | 'sparse' | 'hybrid';
    textSnippet: string;
    assignee: string;
    year: string;
    level: 'Patent' | 'Claim' | 'Limitation';
}

export enum SearchMode {
    PRIOR_ART = 'Prior Art',
    INFRINGEMENT = 'Infringement',
    LANDSCAPE = 'Landscape'
}

export interface SearchResponse {
    answer: string;
    mode: SearchMode;
    evidence: EvidenceChunk[];
}

export interface SearchFilters {
    yearFrom?: string;
    yearTo?: string;
    cpcCodes?: string;
    assignees?: string;
    topK: number;
}

export interface PatentQueryPayload {
    query: string;
    system_description?: string;
    filters?: BackendFilters;
}

export interface BackendFilters {
    cpc_prefixes?: string[];
    year_from?: number;
    year_to?: number;
    assignees?: string[];
}

export interface SearchResponse {
    answer: string;
    mode: SearchMode;
    evidence: EvidenceChunk[];
}