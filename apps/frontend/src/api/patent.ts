import { apiClient } from './client';
import type { QueryRequest, QueryResponse } from '../types';

export const searchPatents = (payload: QueryRequest, signal?: AbortSignal) =>
    apiClient.post<QueryResponse>('/api/v1/query', payload, { signal });