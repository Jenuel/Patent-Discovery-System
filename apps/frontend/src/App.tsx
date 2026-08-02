import { useState } from 'react';
import { Topbar } from './components/Topbar';
import { Landing } from './components/Landing';
import { Results } from './components/Results';
import { EMPTY_COMPOSER } from './constants';
import { usePatentSearch } from './hooks/usePatentSearch';
import type { ComposerState } from './types';

type View = 'landing' | 'results';

function App() {
    const [composer, setComposer] = useState<ComposerState>(EMPTY_COMPOSER);
    const [view, setView] = useState<View>('landing');
    const { status, response, error, submittedQuery, runId, search, cancel, reset } =
        usePatentSearch();

    const run = (next: ComposerState) => {
        if (next.query.trim().length < 3) return;
        setComposer(next);
        setView('results');
        void search(next);
    };

    const editQuery = () => {
        cancel();
        setView('landing');
    };

    const home = () => {
        reset();
        setComposer(EMPTY_COMPOSER);
        setView('landing');
    };

    return (
        <div className="app">
            <Topbar onHome={home} />

            {view === 'results' && status !== 'error' ? (
                <Results
                    // Each run gets a fresh view — nothing carried over from the last.
                    key={runId}
                    query={submittedQuery}
                    response={response}
                    loading={status === 'loading'}
                    onEditQuery={editQuery}
                />
            ) : (
                <Landing
                    composer={composer}
                    onComposerChange={setComposer}
                    onSubmit={() => run(composer)}
                    onExample={(query) => run({ ...composer, query })}
                    busy={status === 'loading'}
                    error={error}
                />
            )}
        </div>
    );
}

export default App;
