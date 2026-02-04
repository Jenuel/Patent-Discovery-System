import React from 'react';
import { ShieldAlert, BookOpenCheck, Zap } from 'lucide-react';
import { FEATURES } from '../../constants';

const Hero: React.FC = () => {
    const getIcon = (iconName: string) => {
        switch (iconName) {
            case 'ShieldAlert': return <ShieldAlert className="w-5 h-5" />;
            case 'Zap': return <Zap className="w-5 h-5" />;
            case 'BookOpenCheck': return <BookOpenCheck className="w-5 h-5" />;
            default: return null;
        }
    };

    const getColorClasses = (color: string) => {
        switch (color) {
            case 'indigo': return 'bg-indigo-50 text-indigo-600';
            case 'rose': return 'bg-rose-50 text-rose-600';
            case 'emerald': return 'bg-emerald-50 text-emerald-600';
            default: return 'bg-slate-50 text-slate-600';
        }
    };

    return (
        <div className="mb-12 animate-in fade-in slide-in-from-top-4 duration-700">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 text-xs font-bold uppercase tracking-widest mb-6 border border-indigo-100">
                <Zap className="w-3 h-3" /> Powered by Advanced RAG Models
            </div>
            <h1 className="text-5xl md:text-6xl font-extrabold text-slate-900 tracking-tight mb-6">
                Next-Generation <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-blue-500">Patent Intelligence</span>
            </h1>
            <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
                Search millions of patents using natural language. Analyze infringement, explore landscapes, and discover prior art with enterprise-grade AI.
            </p>

            <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
                {FEATURES.map((feature) => (
                    <div
                        key={feature.id}
                        className="p-6 rounded-2xl border border-slate-100 bg-white shadow-sm hover:shadow-md transition-shadow"
                    >
                        <div className={`w-10 h-10 ${getColorClasses(feature.color)} rounded-lg flex items-center justify-center mb-4 mx-auto`}>
                            {getIcon(feature.icon)}
                        </div>
                        <h3 className="font-bold text-slate-900 mb-2">{feature.title}</h3>
                        <p className="text-sm text-slate-500">{feature.description}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Hero;
