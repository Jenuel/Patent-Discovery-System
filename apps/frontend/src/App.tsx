import { Header, Footer, Hero, SearchPanel, ResultsView, ErrorMessage } from './components';
import { usePatentSearch } from './hooks';

function App() {
    const { results, isLoading, error, handleSearch } = usePatentSearch();

    return (
        <div className="min-h-screen flex flex-col bg-slate-50">
            <Header />

            <main className="flex-1">
                <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16">
                    {!results && (
                        <div className="text-center">
                            <Hero />
                        </div>
                    )}

                    <SearchPanel onSearch={handleSearch} isLoading={isLoading} />
                </section>

                {error && (
                    <div className="mt-8">
                        <ErrorMessage message={error} />
                    </div>
                )}

                {results && <ResultsView data={results} />}
            </main>

            <Footer />
        </div>
    );
}

export default App;
