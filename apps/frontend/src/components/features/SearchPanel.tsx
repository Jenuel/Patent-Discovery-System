import React, { useState } from 'react';
import { Search, ChevronDown, ChevronUp, SlidersHorizontal, AlertCircle, FileText } from 'lucide-react';
import type { SearchFilters } from '../../types';
import { DEFAULT_FILTERS, FILTER_LIMITS } from '../../constants';

interface SearchPanelProps {
    onSearch: (query: string, systemDescription: string, filters: SearchFilters) => void;
    isLoading: boolean;
}

const SearchPanel: React.FC<SearchPanelProps> = ({ onSearch, isLoading }) => {
    const [query, setQuery] = useState('');
    const [systemDescription, setSystemDescription] = useState('');
    const [isFiltersOpen, setIsFiltersOpen] = useState(false);
    const [showSystemInput, setShowSystemInput] = useState(false);

    const [filters, setFilters] = useState<SearchFilters>({ ...DEFAULT_FILTERS });

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return;
        onSearch(query, systemDescription, filters);
    };

    return (
        <div className="w-full max-w-4xl mx-auto">
            <form onSubmit={handleSearch} className="bg-white shadow-xl shadow-slate-200/50 rounded-2xl border border-slate-200 p-6 md:p-8">
                <div className="relative mb-4">
                    <div className="absolute left-4 top-4 text-slate-400">
                        <Search className="w-6 h-6" />
                    </div>
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Describe an invention, technology area, or ask a patent question..."
                        className="w-full pl-12 pr-4 pt-4 pb-4 min-h-[120px] bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all outline-none resize-none text-slate-800 placeholder:text-slate-400 font-medium text-lg"
                    />
                </div>

                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex items-center space-x-3">
                        <button
                            type="button"
                            onClick={() => setShowSystemInput(!showSystemInput)}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${showSystemInput ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100'}`}
                        >
                            <FileText className="w-4 h-4" />
                            {showSystemInput ? 'Remove System Desc' : 'Add System Description'}
                        </button>
                        <button
                            type="button"
                            onClick={() => setIsFiltersOpen(!isFiltersOpen)}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${isFiltersOpen ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100'}`}
                        >
                            <SlidersHorizontal className="w-4 h-4" />
                            Advanced Filters
                            {isFiltersOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        </button>
                    </div>

                    <button
                        type="submit"
                        disabled={isLoading || !query.trim()}
                        className="bg-indigo-600 text-white px-8 py-3 rounded-xl font-bold text-lg hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-all shadow-lg shadow-indigo-200"
                    >
                        {isLoading ? (
                            <div className="flex items-center gap-2">
                                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Analyzing...
                            </div>
                        ) : 'Discover Patents'}
                    </button>
                </div>

                {showSystemInput && (
                    <div className="mt-6 border-t border-slate-100 pt-6 animate-in slide-in-from-top-2 duration-200">
                        <div className="flex items-center gap-2 mb-3 text-indigo-700 font-bold text-sm uppercase tracking-wide">
                            <AlertCircle className="w-4 h-4" />
                            Infringement Check: System Description
                        </div>
                        <textarea
                            value={systemDescription}
                            onChange={(e) => setSystemDescription(e.target.value)}
                            placeholder="Paste the technical description of the system you want to analyze for potential infringement risks..."
                            className="w-full p-4 min-h-[100px] bg-indigo-50/30 border border-indigo-100 rounded-xl focus:ring-2 focus:ring-indigo-400 outline-none resize-none text-slate-800 placeholder:text-slate-400"
                        />
                    </div>
                )}

                {isFiltersOpen && (
                    <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6 border-t border-slate-100 pt-6 animate-in slide-in-from-top-2 duration-200">
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-slate-500 uppercase">Priority Date Range</label>
                            <div className="flex items-center gap-2">
                                <input
                                    type="text"
                                    placeholder="From (YYYY)"
                                    className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-sm"
                                    value={filters.yearFrom}
                                    onChange={(e) => setFilters({ ...filters, yearFrom: e.target.value })}
                                />
                                <span className="text-slate-400">-</span>
                                <input
                                    type="text"
                                    placeholder="To (YYYY)"
                                    className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-sm"
                                    value={filters.yearTo}
                                    onChange={(e) => setFilters({ ...filters, yearTo: e.target.value })}
                                />
                            </div>
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-slate-500 uppercase">CPC Codes / Class</label>
                            <input
                                type="text"
                                placeholder="e.g. G06N, H04L"
                                className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-sm"
                                value={filters.cpcCodes}
                                onChange={(e) => setFilters({ ...filters, cpcCodes: e.target.value })}
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-slate-500 uppercase">Assignee / Company</label>
                            <input
                                type="text"
                                placeholder="e.g. Google, Apple"
                                className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-sm"
                                value={filters.assignees}
                                onChange={(e) => setFilters({ ...filters, assignees: e.target.value })}
                            />
                        </div>
                        <div className="md:col-span-3 space-y-2">
                            <div className="flex justify-between items-center">
                                <label className="text-xs font-bold text-slate-500 uppercase">Top K Candidates</label>
                                <span className="text-xs font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">{filters.topK} Patents</span>
                            </div>
                            <input
                                type="range"
                                min={FILTER_LIMITS.topK.min}
                                max={FILTER_LIMITS.topK.max}
                                step={FILTER_LIMITS.topK.step}
                                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                                value={filters.topK}
                                onChange={(e) => setFilters({ ...filters, topK: parseInt(e.target.value) })}
                            />
                        </div>
                    </div>
                )}
            </form>
        </div>
    );
};

export default SearchPanel;
