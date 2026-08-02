import { useState } from 'react';
import { ArrowRight, Paperclip, SlidersHorizontal } from 'lucide-react';
import { COMPOSER_PLACEHOLDER, EXAMPLE_QUERIES, SEARCH_CTA } from '../constants';
import type { ComposerState } from '../types';

interface ComposerProps {
    value: ComposerState;
    onChange: (next: ComposerState) => void;
    onSubmit: () => void;
    onExample: (query: string) => void;
    busy: boolean;
}

/** The page's one input. The two tools under the rule open the fields the
 *  backend accepts: system_description, and the filters object. */
export const Composer = ({ value, onChange, onSubmit, onExample, busy }: ComposerProps) => {
    const [showDescription, setShowDescription] = useState(false);
    const [showConstraints, setShowConstraints] = useState(false);

    const set = <K extends keyof ComposerState>(key: K, next: ComposerState[K]) =>
        onChange({ ...value, [key]: next });

    const canSubmit = value.query.trim().length >= 3 && !busy;

    const submit = () => {
        if (canSubmit) onSubmit();
    };

    return (
        <div className="composerWrap">
            <div className="composer">
                <label className="srOnly" htmlFor="composer-query">
                    Describe your invention
                </label>
                <textarea
                    id="composer-query"
                    className="composer__input"
                    rows={3}
                    placeholder={COMPOSER_PLACEHOLDER}
                    value={value.query}
                    onChange={(e) => set('query', e.target.value)}
                    onKeyDown={(e) => {
                        // Enter searches; Shift+Enter keeps the newline.
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            submit();
                        }
                    }}
                />

                <div className="composer__bar">
                    <div className="composer__tools">
                        <button
                            type="button"
                            className="ctl"
                            aria-pressed={showDescription}
                            onClick={() => setShowDescription((open) => !open)}
                        >
                            <Paperclip size={12} strokeWidth={2} aria-hidden="true" /> Attach system
                            description
                        </button>
                        <button
                            type="button"
                            className="ctl"
                            aria-pressed={showConstraints}
                            onClick={() => setShowConstraints((open) => !open)}
                        >
                            <SlidersHorizontal size={12} strokeWidth={2} aria-hidden="true" /> Dates ·
                            CPC
                        </button>
                    </div>
                    <button
                        type="button"
                        className="ctl-primary"
                        onClick={submit}
                        disabled={!canSubmit}
                    >
                        {busy ? 'Searching…' : SEARCH_CTA}{' '}
                        <ArrowRight size={13} strokeWidth={2.4} aria-hidden="true" />
                    </button>
                </div>

                {showDescription && (
                    <div className="composer__drawer">
                        <div className="composer__drawerLabel">SYSTEM DESCRIPTION</div>
                        <div className="composer__field">
                            <label htmlFor="composer-system">
                                system_description — supplying this runs the query as infringement
                            </label>
                            <textarea
                                id="composer-system"
                                className="input"
                                rows={4}
                                placeholder="How your product works, element by element…"
                                value={value.systemDescription}
                                onChange={(e) => set('systemDescription', e.target.value)}
                            />
                        </div>
                    </div>
                )}

                {showConstraints && (
                    <div className="composer__drawer">
                        <div className="composer__drawerLabel">CONSTRAIN THE RETRIEVAL</div>
                        <div className="composer__grid">
                            <div className="composer__field">
                                <label htmlFor="composer-cpc">cpc_prefixes</label>
                                <input
                                    id="composer-cpc"
                                    className="input"
                                    placeholder="G06N, G06V"
                                    value={value.cpcCodes}
                                    onChange={(e) => set('cpcCodes', e.target.value)}
                                />
                            </div>
                            <div className="composer__field">
                                <label htmlFor="composer-from">year_from</label>
                                <input
                                    id="composer-from"
                                    className="input"
                                    inputMode="numeric"
                                    placeholder="2010"
                                    value={value.yearFrom}
                                    onChange={(e) => set('yearFrom', e.target.value)}
                                />
                            </div>
                            <div className="composer__field">
                                <label htmlFor="composer-to">year_to</label>
                                <input
                                    id="composer-to"
                                    className="input"
                                    inputMode="numeric"
                                    placeholder="2024"
                                    value={value.yearTo}
                                    onChange={(e) => set('yearTo', e.target.value)}
                                />
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <div className="tryRow">
                <span className="tryRow__label">Try:</span>
                {EXAMPLE_QUERIES.map((example) => (
                    <button
                        key={example}
                        type="button"
                        className="ctl-tint"
                        onClick={() => onExample(example)}
                        disabled={busy}
                    >
                        {example}
                    </button>
                ))}
            </div>
        </div>
    );
};
