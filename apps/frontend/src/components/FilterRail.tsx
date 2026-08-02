import type { CpcFacet } from '../lib/evidence';
import type { ResultFilters } from '../types';

interface FilterRailProps {
    filters: ResultFilters;
    levels: string[];
    sources: string[];
    facets: CpcFacet[];
    onChange: (next: ResultFilters) => void;
    onReset: () => void;
}

const Chips = ({
    legend,
    options,
    active,
    onPick,
}: {
    legend: string;
    options: string[];
    active: string;
    onPick: (value: string) => void;
}) => (
    <>
        <div className="filters__group">{legend}</div>
        <div className="filters__chips">
            {options.map((option) => (
                <button
                    key={option}
                    type="button"
                    className={`chip${active === option ? ' is-on' : ''}`}
                    aria-pressed={active === option}
                    onClick={() => onPick(option)}
                >
                    {option}
                </button>
            ))}
        </div>
    </>
);

export const FilterRail = ({
    filters,
    levels,
    sources,
    facets,
    onChange,
    onReset,
}: FilterRailProps) => (
    <aside className="filters" aria-label="Narrow these results">
        <div className="filters__head">
            <span className="filters__legend">FILTERS</span>
            <button type="button" className="linkBtn" onClick={onReset}>
                Reset
            </button>
        </div>

        <Chips
            legend="level"
            options={levels}
            active={filters.level}
            onPick={(level) => onChange({ ...filters, level })}
        />
        <Chips
            legend="source"
            options={sources}
            active={filters.source}
            onPick={(source) => onChange({ ...filters, source })}
        />

        <div className="filters__scoreRow">
            <label className="filters__scoreLabel" htmlFor="filter-score">
                score ≥
            </label>
            <span className="filters__scoreValue">{filters.minScore.toFixed(2)}</span>
        </div>
        <input
            id="filter-score"
            className="filters__range"
            type="range"
            min={0}
            max={0.95}
            step={0.05}
            value={filters.minScore}
            onChange={(e) => onChange({ ...filters, minScore: Number.parseFloat(e.target.value) })}
        />

        <div className="filters__rule" />

        <div className="filters__legend filters__legend--facets">CPC IN RESULTS</div>
        <div className="filters__facets">
            {facets.length === 0 && (
                <span className="filters__empty">No CPC codes on these chunks.</span>
            )}
            {facets.map((facet) => {
                const on = filters.cpcPrefix === facet.prefix;
                return (
                    <button
                        key={facet.label}
                        type="button"
                        className={`filters__facet${on ? ' is-on' : ''}`}
                        aria-pressed={on}
                        onClick={() => onChange({ ...filters, cpcPrefix: on ? null : facet.prefix })}
                    >
                        <span>{facet.label}</span>
                        <span className="filters__facetCount">{facet.count}</span>
                    </button>
                );
            })}
        </div>
    </aside>
);
