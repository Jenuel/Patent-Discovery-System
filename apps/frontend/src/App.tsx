import { Header, Footer, Hero, SearchPanel, ResultsView, ErrorMessage } from './components';
import { usePatentSearch } from './hooks/usePatentSearch';

function App() {
    const { status, response, error, search, cancel } = usePatentSearch();

    return (
        <div className="min-h-screen flex flex-col bg-slate-50">
            <Header />

            <main className="flex-1">
                <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16">
                    {!response && (
                        <div className="text-center">
                            <Hero />
                        </div>
                    )}

                    <SearchPanel onSearch={search} onCancel={cancel} isLoading={status === 'loading'} />
                </section>

                {error && (
                    <div className="mt-8">
                        <ErrorMessage message={error} />
                    </div>
                )}

                {response && <ResultsView data={response} />}
            </main>

            <Footer />
        </div>
    );
}

export default App;
