import Topbar from './components/Topbar';
import Landing from './components/Landing';
import Composer from './components/Composer';
import Results from './components/Results';
import ErrorMessage from './components/states';
import { usePatentSearch } from './hooks/usePatentSearch';

function App() {
    const { status, response, error, search, cancel } = usePatentSearch();

    return (
        <div className="min-h-screen flex flex-col bg-slate-50">
            <Topbar />

            <main className="flex-1">
                <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16">
                    {!response && (
                        <div className="text-center">
                            <Landing />
                        </div>
                    )}

                    <Composer onSearch={search} onCancel={cancel} isLoading={status === 'loading'} />
                </section>

                {error && (
                    <div className="mt-8">
                        <ErrorMessage message={error} />
                    </div>
                )}

                {response && <Results data={response} />}
            </main>
        </div>
    );
}

export default App;
