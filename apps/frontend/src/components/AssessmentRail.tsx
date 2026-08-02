import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Citation } from '../lib/citations';

interface AssessmentRailProps {
    answer: string;
    citations: Citation[];
    loading: boolean;
    onJump: (chunkId: string) => void;
}

/** The answer, and the evidence it leans on. The citation row is pinned to the
 *  foot rather than sitting under the copy: a real answer runs to several
 *  screens, and a citation you have to scroll to find is no affordance at all. */
export const AssessmentRail = ({ answer, citations, loading, onJump }: AssessmentRailProps) => (
    <section className="assessment" aria-label="Assessment">
        <div className="assessment__head">
            <span className="assessment__title">Assessment</span>
            <span className="assessment__tag">answer</span>
        </div>

        <div className="assessment__body">
            {loading ? (
                <div className="assessment__prose">
                    <div className="skeleton__line w-lg" />
                    <div className="skeleton__line w-lg" />
                    <div className="skeleton__line w-md" />
                    <div className="skeleton__line w-sm" />
                </div>
            ) : (
                <div className="assessment__prose">
                    {answer.trim() ? (
                        <Markdown remarkPlugins={[remarkGfm]}>{answer}</Markdown>
                    ) : (
                        <p>The response carried no answer text.</p>
                    )}
                </div>
            )}
        </div>

        {!loading && citations.length > 0 && (
            <div className="assessment__foot">
                <div className="assessment__footLabel">CITED IN THIS ANSWER</div>
                <div className="assessment__cites">
                    {citations.map((citation) => (
                        <button
                            type="button"
                            className="cite"
                            key={citation.ordinal}
                            onClick={() => onJump(citation.chunkId)}
                            title={`Go to ${citation.patentId} — ${citation.chunkId}`}
                        >
                            {citation.label}
                        </button>
                    ))}
                </div>
            </div>
        )}
    </section>
);
