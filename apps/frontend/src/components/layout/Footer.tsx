import React from 'react';
import { ShieldAlert } from 'lucide-react';
import { APP_CONFIG, FOOTER_LINKS } from '../../constants';

const Footer: React.FC = () => {
    return (
        <footer className="bg-white border-t border-slate-200 py-12 mt-20">
            <div className="max-w-7xl mx-auto px-4 grid grid-cols-1 md:grid-cols-4 gap-8">
                <div className="col-span-2">
                    <div className="flex items-center space-x-2 mb-4">
                        <div className="bg-slate-900 p-1 rounded-md">
                            <ShieldAlert className="h-4 w-4 text-white" />
                        </div>
                        <span className="text-lg font-bold text-slate-900 tracking-tight">
                            {APP_CONFIG.name}
                        </span>
                    </div>
                    <p className="text-slate-500 text-sm max-w-sm mb-6">
                        {APP_CONFIG.description}
                    </p>
                    <div className="flex space-x-4">
                        <div className="w-8 h-8 bg-slate-100 rounded-full"></div>
                        <div className="w-8 h-8 bg-slate-100 rounded-full"></div>
                        <div className="w-8 h-8 bg-slate-100 rounded-full"></div>
                    </div>
                </div>

                <div>
                    <h4 className="font-bold text-slate-900 mb-4">Platform</h4>
                    <ul className="space-y-2 text-sm text-slate-500">
                        {FOOTER_LINKS.platform.map((link) => (
                            <li key={link.label}>
                                <a href={link.href} className="hover:text-indigo-600">
                                    {link.label}
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>

                <div>
                    <h4 className="font-bold text-slate-900 mb-4">Company</h4>
                    <ul className="space-y-2 text-sm text-slate-500">
                        {FOOTER_LINKS.company.map((link) => (
                            <li key={link.label}>
                                <a href={link.href} className="hover:text-indigo-600">
                                    {link.label}
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>
            </div>

            <div className="max-w-7xl mx-auto px-4 pt-8 mt-8 border-t border-slate-100 flex flex-col md:flex-row justify-between items-center text-xs text-slate-400 font-medium">
                <p>© {APP_CONFIG.year} {APP_CONFIG.company} All rights reserved.</p>
                <div className="flex space-x-6 mt-4 md:mt-0">
                    {FOOTER_LINKS.legal.map((link) => (
                        <a key={link.label} href={link.href} className="hover:text-slate-600">
                            {link.label}
                        </a>
                    ))}
                </div>
            </div>
        </footer>
    );
};

export default Footer;
