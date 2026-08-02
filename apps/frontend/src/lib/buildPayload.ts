import type { ComposerState, QueryFilters, QueryRequest } from '../types';

const toYear = (value: string): number | undefined => {
    const n = Number.parseInt(value.trim(), 10);
    return Number.isFinite(n) ? n : undefined;
};

/** Composer state → the request body `POST /api/v1/query` validates against. */
export const buildPayload = (composer: ComposerState): QueryRequest => {
    const payload: QueryRequest = { query: composer.query.trim() };

    const systemDescription = composer.systemDescription.trim();
    if (systemDescription) payload.system_description = systemDescription;

    const filters: QueryFilters = {};

    const prefixes = composer.cpcCodes
        .split(',')
        .map((code) => code.trim())
        .filter(Boolean);
    if (prefixes.length) filters.cpc_prefixes = prefixes;

    const yearFrom = toYear(composer.yearFrom);
    if (yearFrom !== undefined) filters.year_from = yearFrom;

    const yearTo = toYear(composer.yearTo);
    if (yearTo !== undefined) filters.year_to = yearTo;

    if (Object.keys(filters).length) payload.filters = filters;

    return payload;
};
