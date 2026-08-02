import { ArrowUpRight } from 'lucide-react';
import { PATENT_URL } from '../constants';
import {
    barWidth,
    formatScore,
    isClaimLevel,
    isReranked,
    levelLabel,
    readAssignee,
    readCpcCodes,
    readYear,
} from '../lib/evidence';
import type { EvidenceItem } from '../types';

const CPC_VISIBLE = 3;

interface EvidenceRowProps {
    item: EvidenceItem;
    /** 1-based position in the *unfiltered* evidence list — citations use it. */
    ordinal: number;
    active: boolean;
    onSelect: () => void;
}

export const EvidenceRow = ({ item, ordinal, active, onSelect }: EvidenceRowProps) => {
    const assignee = readAssignee(item);
    const year = readYear(item);
    const cpc = readCpcCodes(item);
    const shownCpc = active ? cpc : cpc.slice(0, CPC_VISIBLE);
    const hiddenCpc = cpc.length - shownCpc.length;
    const claim = isClaimLevel(item);

    return (
        <div
            // The citation jump finds its row by this attribute.
            data-chunk-id={item.chunk_id}
            // Selecting a row unclamps its text: the design clamps to two lines,
            // which for a claim-level chunk hides the language the reader came for.
            className={`row${active ? ' is-active is-expanded' : ''}`}
            onClick={onSelect}
        >
            <span className="row__n">[{ordinal}]</span>

            <div className="row__main">
                <div className="row__meta">
                    <span className="row__pid">{item.patent_id || '—'}</span>
                    <span className={`badge badge-level${claim ? ' is-claim' : ''}`}>
                        {levelLabel(item)}
                    </span>
                    <span className={`badge badge-source${isReranked(item) ? ' is-reranked' : ''}`}>
                        {item.source}
                    </span>
                    <span className="row__chunk">{item.chunk_id}</span>
                </div>

                <button type="button" className="row__title" onClick={onSelect}>
                    {item.title || 'Untitled chunk'}
                </button>

                {(assignee || year || cpc.length > 0) && (
                    <div className="row__facts">
                        {assignee && <span className="row__fact">{assignee}</span>}
                        {assignee && year && <span className="row__dot">·</span>}
                        {year && <span className="row__fact">{year}</span>}
                        {shownCpc.map((code) => (
                            <span className="row__cpc" key={code}>
                                {code}
                            </span>
                        ))}
                        {hiddenCpc > 0 && <span className="row__cpc row__cpcMore">+{hiddenCpc}</span>}
                    </div>
                )}

                <div className="row__text">{item.text}</div>
            </div>

            <div className="row__side">
                <span className="row__score">{formatScore(item.score)}</span>
                <div
                    className="row__track"
                    role="img"
                    aria-label={`Retrieval score ${formatScore(item.score)}`}
                >
                    <div className="row__bar" style={{ width: barWidth(item.score) }} />
                </div>
                <a
                    className="row__open"
                    href={PATENT_URL(item.patent_id)}
                    target="_blank"
                    rel="noreferrer"
                    onClick={(e) => e.stopPropagation()}
                >
                    Open patent <ArrowUpRight size={11} strokeWidth={2.4} aria-hidden="true" />
                </a>
            </div>
        </div>
    );
};
