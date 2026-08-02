import { CAPABILITIES, HERO, STATS } from '../constants';

export const Landing = () => (
    <div className="landing">
        <div className="hero">
            <div>
                <div className="hero__kicker">{HERO.kicker}</div>
                <h1 className="hero__title">{HERO.title}</h1>
                <div className="hero__lede">{HERO.lede}</div>
            </div>

            <div className="panel">
                <div className="panel__head">WHAT YOU GET BACK</div>
                {CAPABILITIES.map((capability) => (
                    <div className="panel__row" key={capability.title}>
                        <div className="panel__rowTitle">{capability.title}</div>
                        <div className="panel__rowBody">{capability.body}</div>
                    </div>
                ))}
            </div>
        </div>

        <div className="stats">
            {STATS.map((stat) => (
                <div className="stats__cell" key={stat.value}>
                    <div className="stats__value">{stat.value}</div>
                    <div className="stats__label">{stat.label}</div>
                </div>
            ))}
        </div>
    </div>
);
