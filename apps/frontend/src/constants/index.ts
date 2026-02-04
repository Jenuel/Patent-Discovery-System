export const APP_CONFIG = {
    name: 'PatentDiscovery',
    description: 'Empowering attorneys and researchers with state-of-the-art semantic search technology for the modern patent landscape.',
    year: 2024,
    company: 'Patent Discovery Systems Inc.'
} as const;

export const DEFAULT_FILTERS = {
    yearFrom: '',
    yearTo: '',
    cpcCodes: '',
    assignees: '',
    topK: 30
} as const;

export const FILTER_LIMITS = {
    topK: {
        min: 10,
        max: 100,
        step: 5
    }
} as const;

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

export const FOOTER_LINKS = {
    platform: [
        { label: 'Search Engine', href: '#' },
        { label: 'Claim Mapping', href: '#' },
        { label: 'Portfolio Health', href: '#' },
        { label: 'API Documentation', href: '#' }
    ],
    company: [
        { label: 'About Us', href: '#' },
        { label: 'Legal Info', href: '#' },
        { label: 'Security', href: '#' },
        { label: 'Support', href: '#' }
    ],
    legal: [
        { label: 'Privacy Policy', href: '#' },
        { label: 'Terms of Service', href: '#' },
        { label: 'Cookie Settings', href: '#' }
    ]
} as const;
