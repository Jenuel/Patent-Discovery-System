import { useState } from 'react';
import { Topbar } from './components/Topbar';
import { Landing } from './components/Landing';
import Results from './components/Results';
import { EMPTY_COMPOSER } from './constants';
import { usePatentSearch } from './hooks/usePatentSearch';
import type { ComposerState } from './types';

function App() {
    const [composer, setComposer] = useState<ComposerState>(EMPTY_COMPOSER);
    const { status, response, error, search, reset } = usePatentSearch();

    const run = (next: ComposerState) => {
        if (next.query.trim().length < 3) return;
        setComposer(next);
        void search(next);
    };

    const home = () => {
        reset();
        setComposer(EMPTY_COMPOSER);
    };

    return (
        <div className="app">
            <Topbar onHome={home} />

            {response ? (
                <Results data={response} />
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
