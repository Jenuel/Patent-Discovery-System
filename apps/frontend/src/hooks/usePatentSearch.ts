import axios from 'axios';
import { useCallback, useRef, useState } from 'react';
import { searchPatents } from '../api/patent';
import { buildPayload } from '../lib/buildPayload';
import type { ComposerState, QueryResponse } from '../types';

export type SearchStatus = 'idle' | 'loading' | 'done' | 'error';

const messageFor = (err: unknown): string => {
    if (axios.isAxiosError(err)) {
        if (err.code === 'ECONNABORTED') {
            return 'The search timed out after 2 minutes. Narrow the query, or add a CPC or date filter.';
        }
        const detail = (err.response?.data as { detail?: unknown } | undefined)?.detail;
        if (typeof detail === 'string') return detail;
        if (Array.isArray(detail)) {
            return detail
                .map((d) => (typeof d === 'object' && d && 'msg' in d ? String(d.msg) : String(d)))
                .join('; ');
        }
        if (!err.response) return 'Could not reach the API. Is the backend running on port 8000?';
        return err.message;
    }
    return err instanceof Error ? err.message : 'The query failed.';
};

export const usePatentSearch = () => {
    const [status, setStatus] = useState<SearchStatus>('idle');
    const [response, setResponse] = useState<QueryResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    /** The query the visible results belong to, not what the composer holds now. */
    const [submittedQuery, setSubmittedQuery] = useState('');
    /** Increments per search, so callers can key the results view on it. */
    const [runId, setRunId] = useState(0);
    const abortRef = useRef<AbortController | null>(null);

    const search = useCallback(async (composer: ComposerState) => {
        abortRef.current?.abort();
        const controller = new AbortController();
        abortRef.current = controller;

        setStatus('loading');
        setError(null);
        setSubmittedQuery(composer.query.trim());
        setRunId((id) => id + 1);

        try {
            const payload = buildPayload(composer);
            const { data } = await searchPatents(payload, controller.signal);
            setResponse(data);
            setStatus('done');
        } catch (err) {
            if (axios.isCancel(err)) return; // superseded by a newer search
            setResponse(null);
            setError(messageFor(err));
            setStatus('error');
        }
    }, []);

    const cancel = useCallback(() => {
        abortRef.current?.abort();
        setStatus(response ? 'done' : 'idle');
    }, [response]);

    const reset = useCallback(() => {
        abortRef.current?.abort();
        setResponse(null);
        setError(null);
        setStatus('idle');
    }, []);

    return { status, response, error, submittedQuery, runId, search, cancel, reset };
};
