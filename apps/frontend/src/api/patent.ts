import { apiClient } from './client';
import type { PatentQueryPayload, SearchResponse } from '../types';

export const searchPatents = (payload: PatentQueryPayload, signal?: AbortSignal) =>
    apiClient.post<SearchResponse>('/api/v1/query', payload, { signal });