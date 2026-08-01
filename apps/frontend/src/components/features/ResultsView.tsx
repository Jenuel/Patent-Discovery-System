import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { SearchResponse } from '../../types';
import { SearchMode } from '../../types';
import EvidenceCard from '../common/EvidenceCard';
import { Sparkles, BrainCircuit, History, ArrowRight, SearchX } from 'lucide-react';

interface ResultsViewProps {
    data: SearchResponse;
}

const ResultsView: React.FC<ResultsViewProps> = ({ data }) => {
    // The prompt numbers evidence [1..n] and asks the model to cite those
    // numbers, so the answer's citations map straight onto card positions.
    const citedIndexes = React.useMemo(() => {
        const found = new Set<number>();
        for (const match of data.answer.matchAll(/\[(\d+)\]/g)) {
            const n = Number(match[1]);
            if (n >= 1 && n <= data.evidence.length) found.add(n);
        }
        return found;
    }, [data.answer, data.evidence.length]);

    const getModeIcon = (mode: SearchMode) => {
        switch (mode) {
            case SearchMode.INFRINGEMENT: return <BrainCircuit className="w-5 h-5" />;
            case SearchMode.LANDSCAPE: return <History className="w-5 h-5" />;
            default: return <Sparkles className="w-5 h-5" />;
        }
    };

    const getModeColor = (mode: SearchMode) => {
        switch (mode) {
            case SearchMode.INFRINGEMENT: return 'bg-rose-50 text-rose-700 border-rose-100';
            case SearchMode.LANDSCAPE: return 'bg-amber-50 text-amber-700 border-amber-100';
            default: return 'bg-emerald-50 text-emerald-700 border-emerald-100';
        }
    };

    return (
        <div className="max-w-7xl mx-auto px-4 py-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                <div className="lg:col-span-5 space-y-6">
                    <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-4">
                            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-bold uppercase tracking-wider ${getModeColor(data.mode)}`}>
                                {getModeIcon(data.mode)}
                                {data.mode} Analysis
                            </div>
                        </div>

                        <h2 className="text-2xl font-bold text-slate-900 mb-6 flex items-center gap-3">
                            <div className="p-2 bg-indigo-600 rounded-lg">
                                <Sparkles className="w-5 h-5 text-white" />
                            </div>
                            AI Patent Intelligence
                        </h2>
                        <div className="prose prose-slate prose-lg max-w-none prose-headings:font-bold prose-strong:text-slate-900 prose-a:text-indigo-600">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {data.answer}
                            </ReactMarkdown>
                        </div>

                        {citedIndexes.size > 0 && (
                            <div className="mt-8 pt-8 border-t border-slate-100">
                                <h4 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Evidence Cited Above</h4>
                                <ul className="space-y-3">
                                    {[...citedIndexes].sort((a, b) => a - b).map((n) => (
                                        <li key={n} className="flex items-start gap-3 text-sm text-slate-600">
                                            <ArrowRight className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" />
                                            <span>
                                                <span className="font-mono text-xs text-indigo-600">[{n}]</span>{' '}
                                                <span className="font-semibold text-slate-900">{data.evidence[n - 1].patentId}</span>
                                                {' — '}{data.evidence[n - 1].title}
                                            </span>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </div>

                <div className="lg:col-span-7">
                    <div className="mb-6 flex items-center justify-between">
                        <h3 className="text-lg font-bold text-slate-800">Citing Evidence & Relevant Prior Art</h3>
                        <span className="text-xs font-bold text-slate-400 bg-slate-100 px-2.5 py-1 rounded-full uppercase">
                            {citedIndexes.size} cited of {data.evidence.length}
                        </span>
                    </div>

                    {data.evidence.length === 0 ? (
                        <div className="border-2 border-dashed border-slate-200 rounded-xl p-10 text-center">
                            <SearchX className="w-8 h-8 text-slate-300 mx-auto mb-3" />
                            <h4 className="font-semibold text-slate-800 mb-1">No matching evidence</h4>
                            <p className="text-sm text-slate-500 max-w-sm mx-auto">
                                Nothing in the corpus matched this query. Widening or clearing the
                                CPC and year filters is usually what brings results back.
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {data.evidence.map((chunk, index) => (
                                <EvidenceCard
                                    key={chunk.patentId + index}
                                    evidence={chunk}
                                    rank={index + 1}
                                    isCited={citedIndexes.has(index + 1)}
                                />
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ResultsView;
