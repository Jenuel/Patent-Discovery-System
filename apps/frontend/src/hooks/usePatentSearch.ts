import { useState, useRef } from 'react';
import { searchPatents } from '../api/patent.ts';
import { buildPayload } from '../lib/buildPayload.ts';
import type { ComposerState, QueryResponse } from '../types';

export const usePatentSearch = () => {
    const [results, setResults] = useState<QueryResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    const handleSearch = async (composer: ComposerState) => {
        abortRef.current?.abort();
        abortRef.current = new AbortController();

        setIsLoading(true);
        setError(null);

        try {
            const payload = buildPayload(composer);
            const response = await searchPatents(payload, abortRef.current.signal);
            setResults(response.data);
        } catch (err: any) {
            if (err.name === 'CanceledError') return; // ignore aborted requests
            console.error('Search error:', err);
            if (err.code === 'ECONNABORTED') {
                setError('The search timed out after 2 minutes. Try narrowing the query or applying a CPC filter.');
            } else {
                setError(err.response?.data?.detail || err.message || 'An error occurred while processing your patent query.');
            }
        } finally {
            setIsLoading(false);
        }
    };

    const cancelSearch = () => {
        abortRef.current?.abort();
        setIsLoading(false);
    };

    const clearResults = () => {
        setResults(null);
        setError(null);
    };

    return { results, isLoading, error, handleSearch, cancelSearch, clearResults };
};