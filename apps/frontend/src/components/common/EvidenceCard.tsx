import React from 'react';
import { ExternalLink, Layers, Hash, Gauge } from 'lucide-react';
import type { EvidenceChunk } from '../../types';

interface EvidenceCardProps {
    evidence: EvidenceChunk;
    rank: number;
}

const EvidenceCard: React.FC<EvidenceCardProps> = ({ evidence, rank }) => {
    const getRankStyle = (position: number) =>
        position <= 3
            ? 'text-indigo-700 bg-indigo-50 border-indigo-100'
            : 'text-slate-500 bg-slate-50 border-slate-100';

    const getSourceBadge = (source: string) => {
        switch (source) {
            case 'hybrid': return 'bg-indigo-50 text-indigo-700 border-indigo-100';
            case 'dense': return 'bg-blue-50 text-blue-700 border-blue-100';
            default: return 'bg-slate-50 text-slate-700 border-slate-100';
        }
    };

    return (
        <div className="bg-white border border-slate-200 rounded-xl p-5 hover:shadow-md transition-shadow group">
            <div className="flex justify-between items-start mb-3">
                <div className="flex flex-col">
                    <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-bold text-indigo-600 uppercase tracking-wider">{evidence.patentId}</span>
                        <span className={`text-[10px] px-2 py-0.5 rounded-full border font-medium uppercase ${getRankStyle(rank)}`}>
                            Rank #{rank}
                        </span>
                    </div>
                    <h3 className="text-md font-semibold text-slate-900 group-hover:text-indigo-600 transition-colors">
                        {evidence.title}
                    </h3>
                </div>
                <a
                    href={`https://patents.google.com/patent/${evidence.patentId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-all"
                >
                    <ExternalLink className="w-4 h-4" />
                </a>
            </div>

            <div className="mb-4">
                <p className="text-sm text-slate-600 leading-relaxed italic border-l-2 border-slate-200 pl-4 py-1">
                    "{evidence.textSnippet}"
                </p>
            </div>

            <div className="flex flex-wrap gap-2 items-center text-[11px] font-medium text-slate-500 border-t border-slate-100 pt-3">
                <div className="flex items-center gap-1 mr-3">
                    <Layers className="w-3.5 h-3.5" />
                    <span>Level: <span className="text-slate-900">{evidence.level}</span></span>
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
                    <span>Source: <span className={`px-1.5 rounded border uppercase text-[10px] ${getSourceBadge(evidence.sourceType)}`}>{evidence.sourceType}</span></span>
                </div>
                <div className="ml-auto">
                    <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded">Priority: {evidence.year}</span>
                </div>
            </div>
        </div>
    );
};

export default EvidenceCard;
