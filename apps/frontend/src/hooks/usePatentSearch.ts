import { useState, useRef } from 'react';
import { searchPatents } from '../api/patent.ts';
import { buildPayload } from '../lib/patents/buildPayload.ts';
import { mapSearchResponse } from '../lib/patents/mapResponse.ts';
import type { SearchResponse, SearchFilters } from '../types';

export const usePatentSearch = () => {
    const [results, setResults] = useState<SearchResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    const handleSearch = async (query: string, systemDescription: string, filters: SearchFilters) => {
        abortRef.current?.abort();
        abortRef.current = new AbortController();

        setIsLoading(true);
        setError(null);

        try {
            const payload = buildPayload(query, systemDescription, filters);
            const response = await searchPatents(payload, abortRef.current.signal);
            setResults(mapSearchResponse(response.data));
        } catch (err: any) {
            if (err.name === 'CanceledError') return; // ignore aborted requests
            console.error('Search error:', err);
            setError(err.response?.data?.detail || err.message || 'An error occurred while processing your patent query.');
        } finally {
            setIsLoading(false);
        }
    };

    const clearResults = () => {
        setResults(null);
        setError(null);
    };

    return { results, isLoading, error, handleSearch, clearResults };
};