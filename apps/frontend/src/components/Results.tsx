import { useCallback, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { AssessmentRail } from './AssessmentRail';
import { EvidenceRow } from './EvidenceRow';
import { FilterRail } from './FilterRail';
import { NoEvidenceState, NoMatchesState, SkeletonList } from './states';
import { EMPTY_RESULT_FILTERS } from '../constants';
import { citationsFor } from '../lib/citations';
import { applyFilters, cpcFacets, levelOptions, modeLabel, sourceOptions } from '../lib/evidence';
import type { QueryResponse, ResultFilters } from '../types';

interface ResultsProps {
    query: string;
    response: QueryResponse | null;
    loading: boolean;
    onEditQuery: () => void;
}

export const Results = ({ query, response, loading, onEditQuery }: ResultsProps) => {
    const [filters, setFilters] = useState<ResultFilters>(EMPTY_RESULT_FILTERS);
    const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
    const listRef = useRef<HTMLDivElement | null>(null);

    const evidence = useMemo(() => response?.evidence ?? [], [response]);
    const visible = useMemo(() => applyFilters(evidence, filters), [evidence, filters]);
    const levels = useMemo(() => levelOptions(evidence), [evidence]);
    const sources = useMemo(() => sourceOptions(evidence), [evidence]);
    const facets = useMemo(() => cpcFacets(evidence), [evidence]);
    const answer = response?.answer ?? '';
    const citations = useMemo(() => citationsFor(answer, evidence), [answer, evidence]);

    /** Ordinals are positions in the *unfiltered* list, so `[3]` keeps meaning
     *  the third chunk of the payload however the rail is narrowed. */
    const ordinals = useMemo(() => {
        const map = new Map<string, number>();
        evidence.forEach((item, index) => map.set(item.chunk_id, index + 1));
        return map;
    }, [evidence]);

    const scrollToRow = useCallback((chunkId: string) => {
        const list = listRef.current;
        if (!list) return;
        const el = list.querySelector<HTMLElement>(`[data-chunk-id="${CSS.escape(chunkId)}"]`);
        if (el) list.scrollTo({ top: Math.max(0, el.offsetTop - 10), behavior: 'smooth' });
    }, []);

    const jumpTo = (chunkId: string) => {
        // A citation must never be a dead control: if the filters currently hide
        // the chunk it points at, clear them. flushSync commits that before the
        // scroll below reads the DOM, so the row is there to scroll to.
        flushSync(() => {
            setActiveChunkId(chunkId);
            if (!visible.some((item) => item.chunk_id === chunkId)) {
                setFilters(EMPTY_RESULT_FILTERS);
            }
        });
        scrollToRow(chunkId);
    };

    return (
        <main className="results">
            <div className="queryBar">
                {response && !loading && (
                    <span className="queryBar__mode">{modeLabel(response.mode)}</span>
                )}
                <span className="queryBar__text" title={query}>
                    {query}
                </span>
                <button type="button" className="queryBar__edit" onClick={onEditQuery}>
                    Edit query
                </button>
            </div>

            <div className="results__body">
                <FilterRail
                    filters={filters}
                    levels={levels}
                    sources={sources}
                    facets={facets}
                    onChange={setFilters}
                    onReset={() => setFilters(EMPTY_RESULT_FILTERS)}
                />

                <div className="evidence">
                    <div className="evidence__head">
                        <div className="evidence__count">
                            <span className="evidence__countN">
                                {loading
                                    ? 'Reading the claims…'
                                    : `${visible.length} of ${evidence.length} evidence chunks`}
                            </span>
                            <span className="evidence__countSub">
                                level · source · score · cpc, straight off the payload
                            </span>
                        </div>
                        <span className="evidence__sort">sorted by score ↓</span>
                    </div>

                    {loading && <SkeletonList />}

                    {!loading && evidence.length === 0 && <NoEvidenceState onEdit={onEditQuery} />}

                    {!loading && evidence.length > 0 && visible.length === 0 && (
                        <NoMatchesState onReset={() => setFilters(EMPTY_RESULT_FILTERS)} />
                    )}

                    {!loading && visible.length > 0 && (
                        <div className="evidence__list" ref={listRef}>
                            {visible.map((item) => (
                                <EvidenceRow
                                    key={item.chunk_id}
                                    item={item}
                                    ordinal={ordinals.get(item.chunk_id) ?? 0}
                                    active={activeChunkId === item.chunk_id}
                                    onSelect={() => setActiveChunkId(item.chunk_id)}
                                />
                            ))}
                        </div>
                    )}
                </div>

                <AssessmentRail
                    answer={answer}
                    citations={citations}
                    loading={loading}
                    onJump={jumpTo}
                />
            </div>
        </main>
    );
};
