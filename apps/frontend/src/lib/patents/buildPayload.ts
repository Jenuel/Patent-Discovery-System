import type { SearchFilters, PatentQueryPayload } from '../../types';

export const buildPayload = (
    query: string,
    systemDescription: string,
    filters: SearchFilters
): PatentQueryPayload => {
    const payload: PatentQueryPayload = { query };

    if (systemDescription.trim()) {
        payload.system_description = systemDescription;
    }

    const backendFilters = {
        ...(filters.cpcCodes && { cpc_prefixes: filters.cpcCodes.split(',').map(s => s.trim()).filter(Boolean) }),
        ...(filters.yearFrom && { year_from: parseInt(filters.yearFrom, 10) }),
        ...(filters.yearTo && { year_to: parseInt(filters.yearTo, 10) }),
        ...(filters.assignees && { assignees: filters.assignees.split(',').map(s => s.trim()).filter(Boolean) }),
    };

    if (Object.keys(backendFilters).length > 0) payload.filters = backendFilters;

    return payload;
};