/** Loading, empty and error states in the system's own language: same rules,
 *  same flush-left setting, ink drained out rather than a different look. */

const SKELETON_ROWS = [0, 1, 2, 3, 4];

export const SkeletonList = () => (
    <div className="evidence__list" aria-busy="true" aria-label="Searching">
        {SKELETON_ROWS.map((row) => (
            <div className="skeleton" key={row}>
                <div className="skeleton__line w-sm" />
                <div className="skeleton__line w-md" />
                <div className="skeleton__line w-lg" />
            </div>
        ))}
    </div>
);

export const NoEvidenceState = ({ onEdit }: { onEdit: () => void }) => (
    <div className="state state-inline">
        <div className="state__kicker">NO EVIDENCE</div>
        <h2 className="state__title">Nothing came back.</h2>
        <p className="state__body">
            The query returned no chunks. Widen the CPC prefixes or the year range, or describe the
            invention in more of its own vocabulary — retrieval is over claim language, not titles.
        </p>
        <div className="state__actions">
            <button type="button" className="ctl" onClick={onEdit}>
                Edit the query
            </button>
        </div>
    </div>
);

export const NoMatchesState = ({ onReset }: { onReset: () => void }) => (
    <div className="state state-inline">
        <div className="state__kicker">FILTERED OUT</div>
        <h2 className="state__title">Every chunk is hidden.</h2>
        <p className="state__body">
            The filters on the left exclude all of the evidence that came back. Reset them to see
            the full result set again.
        </p>
        <div className="state__actions">
            <button type="button" className="ctl" onClick={onReset}>
                Reset filters
            </button>
        </div>
    </div>
);

export const ErrorState = ({ message, onRetry }: { message: string; onRetry: () => void }) => (
    <div className="state">
        <div className="state__kicker is-warn">QUERY FAILED</div>
        <h2 className="state__title">The search did not complete.</h2>
        <p className="state__body">{message}</p>
        <div className="state__actions">
            <button type="button" className="ctl" onClick={onRetry}>
                Back to the query
            </button>
        </div>
    </div>
);
