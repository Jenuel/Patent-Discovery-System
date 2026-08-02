import React from 'react';
import { ExternalLink, Layers, Hash, Gauge } from 'lucide-react';
import type { EvidenceItem } from '../../types';

/** Anything the retriever does not label is patent-level. */
const LEVEL_LABEL: Record<string, string> = { claim: 'Claim', limitation: 'Limitation' };

interface EvidenceCardProps {
    evidence: EvidenceItem;
    /** 1-based position, matching the [n] the LLM is asked to cite. */
    rank: number;
    /** True when the generated answer cites this item's number. */
    isCited: boolean;
}

const EvidenceCard: React.FC<EvidenceCardProps> = ({ evidence, rank, isCited }) => {
    const getSourceBadge = (source: string) => {
        switch (source) {
            case 'reranked': return 'bg-emerald-50 text-emerald-700 border-emerald-100';
            case 'hybrid': return 'bg-indigo-50 text-indigo-700 border-indigo-100';
            case 'dense': return 'bg-blue-50 text-blue-700 border-blue-100';
            case 'sparse': return 'bg-amber-50 text-amber-700 border-amber-100';
            default: return 'bg-slate-50 text-slate-700 border-slate-100';
        }
    };

    return (
        <div
            id={`evidence-${rank}`}
            className={`rounded-xl p-5 transition-shadow group ${
                isCited
                    ? 'bg-white border-2 border-indigo-300 shadow-sm ring-1 ring-indigo-100'
                    : 'bg-white border border-slate-200 hover:shadow-md'
            }`}
        >
            <div className="flex justify-between items-start mb-3">
                <div className="flex flex-col">
                    <div className="flex items-center gap-2 mb-1">
                        <span className="font-mono text-xs text-slate-400" title="Citation number used in the answer above">
                            [{rank}]
                        </span>
                        <span className="text-xs font-bold text-indigo-600 uppercase tracking-wider">{evidence.patent_id}</span>
                        {isCited && (
                            <span className="text-[10px] px-2 py-0.5 rounded-full border font-medium uppercase text-indigo-700 bg-indigo-50 border-indigo-100">
                                Cited
                            </span>
                        )}
                    </div>
                    <h3 className="text-md font-semibold text-slate-900 group-hover:text-indigo-600 transition-colors">
                        {evidence.title ?? 'Unknown Title'}
                    </h3>
                </div>
                <a
                    href={`https://patents.google.com/patent/${evidence.patent_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-all"
                >
                    <ExternalLink className="w-4 h-4" />
                </a>
            </div>

            <div className="mb-4">
                <p className="text-sm text-slate-600 leading-relaxed italic border-l-2 border-slate-200 pl-4 py-1">
                    "{evidence.text}"
                </p>
            </div>

            <div className="flex flex-wrap gap-2 items-center text-[11px] font-medium text-slate-500 border-t border-slate-100 pt-3">
                <div className="flex items-center gap-1 mr-3">
                    <Layers className="w-3.5 h-3.5" />
                    <span>Level: <span className="text-slate-900">{LEVEL_LABEL[evidence.level?.toLowerCase()] ?? 'Patent'}</span></span>
                </div>
                <div
                    className="flex items-center gap-1 mr-3"
                    title="Hybrid dense+sparse fusion score. Comparable within this result set only — not a similarity percentage."
                >
                    <Gauge className="w-3.5 h-3.5" />
                    <span>Score: <span className="text-slate-900 tabular-nums">{evidence.score.toFixed(3)}</span></span>
                </div>
                <div className="flex items-center gap-1 mr-3">
                    <Hash className="w-3.5 h-3.5" />
                    <span>Source: <span className={`px-1.5 rounded border uppercase text-[10px] ${getSourceBadge(evidence.source)}`}>{evidence.source}</span></span>
                </div>
                <div className="ml-auto">
                    <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded">Priority: {evidence.metadata?.year ?? 'Unknown Year'}</span>
                </div>
            </div>
        </div>
    );
};

export default EvidenceCard;
