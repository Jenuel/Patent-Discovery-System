import React from 'react';
import type { SearchResponse } from '../../types';
import { SearchMode } from '../../types';
import EvidenceCard from '../common/EvidenceCard';
import { Sparkles, BrainCircuit, History, ArrowRight } from 'lucide-react';

interface ResultsViewProps {
    data: SearchResponse;
}

const ResultsView: React.FC<ResultsViewProps> = ({ data }) => {
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

                        <div className="prose prose-slate max-w-none">
                            <div className="text-slate-700 leading-relaxed text-lg space-y-4">
                                {data.answer.split('\n').map((para, i) => (
                                    <p key={i}>{para}</p>
                                ))}
                            </div>
                        </div>

                        <div className="mt-8 pt-8 border-t border-slate-100">
                            <h4 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Key Findings</h4>
                            <ul className="space-y-3">
                                {data.evidence.slice(0, 3).map((ev, i) => (
                                    <li key={i} className="flex items-start gap-3 text-sm text-slate-600">
                                        <ArrowRight className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" />
                                        <span>Significant overlap detected with <span className="font-semibold text-slate-900">{ev.patentId}</span> ({ev.assignee})</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>

                <div className="lg:col-span-7">
                    <div className="mb-6 flex items-center justify-between">
                        <h3 className="text-lg font-bold text-slate-800">Citing Evidence & Relevant Prior Art</h3>
                        <span className="text-xs font-bold text-slate-400 bg-slate-100 px-2.5 py-1 rounded-full uppercase">
                            {data.evidence.length} Chunks Retrieved
                        </span>
                    </div>

                    <div className="space-y-4">
                        {data.evidence.map((chunk, index) => (
                            <EvidenceCard key={index} evidence={chunk} />
                        ))}
                    </div>

                    <button className="w-full mt-6 py-4 border-2 border-dashed border-slate-200 rounded-xl text-slate-500 font-semibold hover:bg-slate-50 hover:border-indigo-300 hover:text-indigo-600 transition-all">
                        Load More Results
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ResultsView;
