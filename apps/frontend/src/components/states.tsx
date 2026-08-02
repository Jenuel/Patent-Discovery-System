import React from 'react';
import { ShieldAlert } from 'lucide-react';

interface ErrorMessageProps {
    message: string;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message }) => {
    return (
        <div className="max-w-4xl mx-auto px-4 mb-8">
            <div className="bg-rose-50 border border-rose-100 text-rose-800 p-4 rounded-xl flex items-center gap-3">
                <ShieldAlert className="w-5 h-5 flex-shrink-0" />
                <p className="text-sm font-medium">{message}</p>
            </div>
        </div>
    );
};

export default ErrorMessage;
