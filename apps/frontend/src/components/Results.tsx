import { useCallback, useMemo, useRef, useState } from 'react';
import { AssessmentRail } from './AssessmentRail';
import { EvidenceRow } from './EvidenceRow';
import { NoEvidenceState, SkeletonList } from './states';
import { citationsFor } from '../lib/citations';
import { modeLabel } from '../lib/evidence';
import type { QueryResponse } from '../types';

interface ResultsProps {
    query: string;
    response: QueryResponse | null;
    loading: boolean;
    onEditQuery: () => void;
}

export const Results = ({ query, response, loading, onEditQuery }: ResultsProps) => {
    const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
    const listRef = useRef<HTMLDivElement | null>(null);

    const evidence = useMemo(() => response?.evidence ?? [], [response]);
    const answer = response?.answer ?? '';
    const citations = useMemo(() => citationsFor(answer, evidence), [answer, evidence]);

    const jumpTo = useCallback((chunkId: string) => {
        setActiveChunkId(chunkId);
        const list = listRef.current;
        if (!list) return;
        const el = list.querySelector<HTMLElement>(`[data-chunk-id="${CSS.escape(chunkId)}"]`);
        if (el) list.scrollTo({ top: Math.max(0, el.offsetTop - 10), behavior: 'smooth' });
    }, []);

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
                <div className="evidence">
                    <div className="evidence__head">
                        <div className="evidence__count">
                            <span className="evidence__countN">
                                {loading
                                    ? 'Reading the claims…'
                                    : `${evidence.length} evidence chunks`}
                            </span>
                            <span className="evidence__countSub">
                                level · source · score · cpc, straight off the payload
                            </span>
                        </div>
                        <span className="evidence__sort">sorted by score ↓</span>
                    </div>

                    {loading && <SkeletonList />}

                    {!loading && evidence.length === 0 && <NoEvidenceState onEdit={onEditQuery} />}

                    {!loading && evidence.length > 0 && (
                        <div className="evidence__list" ref={listRef}>
                            {evidence.map((item, index) => (
                                <EvidenceRow
                                    key={item.chunk_id}
                                    item={item}
                                    ordinal={index + 1}
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
