import type { ComposerState } from '../types';

export const APP_NAME = { head: 'Patent', tail: 'Discovery' } as const;

export const EMPTY_COMPOSER: ComposerState = {
    query: '',
    systemDescription: '',
    cpcCodes: '',
    yearFrom: '',
    yearTo: '',
};

export const FEATURES = [
    {
        id: 'prior-art',
        title: 'Prior Art Search',
        description: 'Find blocking patents and similar inventions with semantic understanding.',
        icon: 'ShieldAlert',
        color: 'indigo'
    },
    {
        id: 'infringement',
        title: 'Infringement Risk',
        description: 'Input your system description to detect potential claim overlap automatically.',
        icon: 'Zap',
        color: 'rose'
    },
    {
        id: 'landscape',
        title: 'Landscape Trends',
        description: 'Understand the competitive landscape and key assignees in any technology field.',
        icon: 'BookOpenCheck',
        color: 'emerald'
    }
] as const;

