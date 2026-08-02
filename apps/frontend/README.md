# Patent Discovery System — Frontend

Two screens bound to the real `POST /api/v1/query` contract:

- **Landing.** Flush-left hero, ruled capability list, the composer is the page.
- **Results.** Three columns bound to the response shape: what narrows the
  evidence, the evidence itself, and the answer written from it.

Built with React 19, TypeScript, Vite, and a hand-written CSS design system.

## Running it

```sh
npm install
npm run dev          # http://localhost:5173
```

`/api` is proxied to `http://localhost:8000`, so run the API from `apps/api`
alongside it. Point elsewhere with `VITE_API_URL` in `.env`:

```env
VITE_API_URL=http://localhost:8000
```

```sh
npm run build        # tsc -b && vite build
npm run lint
```

### Docker

`Dockerfile` builds the app and serves `dist/` from nginx, with `/health`
wired up for the healthcheck:

```sh
docker build -t patent-frontend .
docker run -p 8080:80 patent-frontend
```

## The design system

`src/styles/` is three layers, loaded in this order by `src/main.tsx`:

| File | What it is |
| --- | --- |
| `design-system.css` | Modernist — tokens, type scale, primitives. |
| `theme.css` | The `:root` overrides: a cool ground and an indigo accent. |
| `app.css` | The two screens as named classes. Every value is a token; the literals that remain are ones the design fixes (2px rules, the 186px filter rail, the 396px assessment rail). |

Structural rules throughout: 0px radius, 2px dividers between major sections,
Archivo for headings and body, flush-left labels, Lucide icons. Every
interactive element gets a hover tint, a pressed step from the accent ramp, and
a 2px accent focus ring.

## Binding to the payload

Every field on the results screen is a field of `QueryResponse`. `src/types`
keeps backend snake_case (`chunk_id`, `claim_no`, `level`, `source`) on purpose:
what you read is what the API returned.

| On screen | From |
| --- | --- |
| `[n]` ordinal, and the `[n]` citations in the answer | position in `evidence[]` |
| `CLAIM 4` / `PATENT LEVEL` badge | `level` + `claim_no` |
| source badge, tinted when reranked | `source` |
| `0.5143` and the bar | `score` |
| `US20180189327A1::claim::0001` | `chunk_id` |
| assignee · year · CPC tags | `metadata` |
| `PRIOR ART` / `INFRINGEMENT` badge in the query bar | `mode` |

Derived, not returned:

- **Citations.** The orchestrator's answer prompt numbers every evidence item
  and tells the model to cite `[3]` or `[7][12]` after the statement it
  supports, so the brackets in the answer *are* indexes into `evidence[]`.
  They are parsed out and pinned to the foot of the rail as jump buttons.
  Ordinals outside `1…evidence.length` are dropped, so a hallucinated `[47]`
  never becomes a control that goes nowhere. Matching patent numbers in the
  prose is a fallback for a bracket-less answer, not a supplement.
- **CPC facet counts.** One entry per subgroup (`G06N3/*`), collapsed to the
  4-character section (`G06F*`) when a section contributes more than one. The
  corpus stores codes unpunctuated (`G06F1730259`), and with no subgroup
  boundary to read those group at the section. Counts are documents, not codes.
- **Filter chip vocabulary.** `level` and `source` chips are read off the
  response rather than hardcoded, so a result set with no patent-level chunks
  does not offer a dead `patent` chip.

### Everything is keyed by `chunk_id`, not `patent_id`

Claim-level retrieval returns several chunks per patent — a single response
routinely carries claims 1, 8, 9 and 10 of the same filing. Keying on
`patent_id` would make `[3]`, `[7]` and `[12]` all scroll to the same row. Row
identity, selection and citation jumps all use `chunk_id`.

### What the current deployment returns

Worth knowing when the screen looks emptier than expected:

- **Only `claim`-level chunks, only `hybrid` source.** `RERANK_ENABLED` is unset
  in `apps/api/.env`, so nothing is marked `reranked` and the tinted source
  badge never fires.
- **No `assignee`.** HUPD chunks carry `year`, `cpc` and `cpc_prefix` only, so
  the facts line renders as `year · CPC` and drops the segment rather than
  printing a placeholder.
- **Four to six CPC codes per chunk.** Rows show three and a `+N`; selecting the
  row shows the rest.

## Notes

- The composer's two tools open drawers for `system_description` (which switches
  the query to infringement mode) and the `filters` object. There is no
  jurisdiction control: the request schema has `cpc_prefixes`, `year_from` and
  `year_to` and no jurisdiction field.
- The answer is rendered as markdown — it comes back as markdown, so flattening
  it would print `**` and `*` on screen.
- Selecting a row unclamps its text. Chunks clamp to two lines otherwise, which
  for a claim-level chunk hides the language the reader came for.
- `142M filings`, `34 offices` and `40s` in `src/constants/index.ts` describe the
  full-scale target rather than what this deployment indexes. They are isolated
  in that file so they can be retuned in one place.

## Contributing

Follow the guidelines in the root [README](../../README.md).
