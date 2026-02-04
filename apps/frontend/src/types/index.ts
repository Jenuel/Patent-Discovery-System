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
