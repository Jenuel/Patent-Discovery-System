import React from 'react';
import { Shield, Search, BookOpen, BarChart3 } from 'lucide-react';

const Header: React.FC = () => {
    return (
        <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-16">
                    <div className="flex items-center space-x-2">
                        <div className="bg-indigo-600 p-1.5 rounded-lg">
                            <Shield className="h-6 w-6 text-white" />
                        </div>
                        <span className="text-xl font-bold text-slate-900 tracking-tight">
                            Patent<span className="text-indigo-600">Discovery</span>
                        </span>
                    </div>

                    <nav className="hidden md:flex space-x-8">
                        <a href="#" className="text-sm font-medium text-slate-600 hover:text-indigo-600 flex items-center gap-1.5 transition-colors">
                            <Search className="w-4 h-4" /> Search
                        </a>
                        <a href="#" className="text-sm font-medium text-slate-600 hover:text-indigo-600 flex items-center gap-1.5 transition-colors">
                            <BookOpen className="w-4 h-4" /> Portfolios
                        </a>
                        <a href="#" className="text-sm font-medium text-slate-600 hover:text-indigo-600 flex items-center gap-1.5 transition-colors">
                            <BarChart3 className="w-4 h-4" /> Analytics
                        </a>
                    </nav>

                    <div className="flex items-center space-x-4">
                        <button className="text-sm font-medium text-slate-600 hover:text-slate-900">Sign In</button>
                        <button className="bg-slate-900 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-slate-800 transition-all">
                            Try Pro
                        </button>
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;
