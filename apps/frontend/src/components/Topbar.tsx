import { Shield } from 'lucide-react';
import { APP_NAME } from '../constants';

interface TopbarProps {
    onHome: () => void;
}

export const Topbar = ({ onHome }: TopbarProps) => (
    <header className="topbar">
        <div className="topbar__left">
            <button type="button" className="brand" onClick={onHome}>
                <span className="brand__mark">
                    <Shield size={15} strokeWidth={2.2} aria-hidden="true" />
                </span>
                <span className="brand__word">
                    {APP_NAME.head}
                    <em>{APP_NAME.tail}</em>
                </span>
                <span className="srOnly">Start a new search</span>
            </button>
        </div>
    </header>
);
