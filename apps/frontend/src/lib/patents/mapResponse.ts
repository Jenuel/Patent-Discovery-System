import { SearchMode, type SearchResponse } from '../../types';

const mapMode = (mode: string): SearchMode => {
    if (mode === 'infringement') return SearchMode.INFRINGEMENT;
    if (mode === 'landscape') return SearchMode.LANDSCAPE;
    return SearchMode.PRIOR_ART;
};

const mapLevel = (level?: string): 'Claim' | 'Limitation' | 'Patent' => {
    if (level?.toLowerCase() === 'claim') return 'Claim';
    if (level?.toLowerCase() === 'limitation') return 'Limitation';
    return 'Patent';
};

export const mapSearchResponse = (data: any): SearchResponse => ({
    answer: data.answer ?? '',
    mode: mapMode(data.mode as SearchMode),
    evidence: (data.evidence ?? []).map((item: any) => ({
        patentId: item.patent_id ?? '',
        title: item.title ?? 'Unknown Title',
        score: item.score ?? 0,
        sourceType: item.source ?? 'hybrid',
        textSnippet: item.text ?? '',
        assignee: item.metadata?.assignee ?? 'Unknown Assignee',
        year: item.metadata?.year?.toString() ?? 'Unknown Year',
        level: mapLevel(item.level),
    }))
});