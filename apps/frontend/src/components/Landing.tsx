import { Composer } from './Composer';
import { CAPABILITIES, HERO, STATS } from '../constants';
import type { ComposerState } from '../types';

interface LandingProps {
    composer: ComposerState;
    onComposerChange: (next: ComposerState) => void;
    onSubmit: () => void;
    onExample: (query: string) => void;
    busy: boolean;
    error: string | null;
}

export const Landing = ({
    composer,
    onComposerChange,
    onSubmit,
    onExample,
    busy,
    error,
}: LandingProps) => (
    <main className="landing">
        {error && (
            <div className="errorBar" role="alert">
                {error}
            </div>
        )}

        <div className="hero">
            <div>
                <div className="hero__kicker">{HERO.kicker}</div>
                <h1 className="hero__title">{HERO.title}</h1>
                <div className="hero__lede">{HERO.lede}</div>
            </div>

            <div className="panel">
                <div className="panel__head">WHAT YOU GET BACK</div>
                {CAPABILITIES.map((capability) => (
                    <div className="panel__row" key={capability.title}>
                        <div className="panel__rowTitle">{capability.title}</div>
                        <div className="panel__rowBody">{capability.body}</div>
                    </div>
                ))}
            </div>
        </div>

        <Composer
            value={composer}
            onChange={onComposerChange}
            onSubmit={onSubmit}
            onExample={onExample}
            busy={busy}
        />

        <div className="stats">
            {STATS.map((stat) => (
                <div className="stats__cell" key={stat.value}>
                    <div className="stats__value">{stat.value}</div>
                    <div className="stats__label">{stat.label}</div>
                </div>
            ))}
        </div>
    </main>
);
