import { useEffect, useState } from "react";

type Row = {
  item?: string; lane_role?: number; scenario?: string; is_radiant?: boolean; region?: number;
  time_minutes?: number; games: number; wins: number; win_rate_pct: number | null; small_sample: boolean;
};
type Dataset = { error?: string; rows?: Row[]; fetched_at?: string; stale?: boolean; limitation?: string };
type Response = { hero: string; scenarios: Record<string, Dataset> };
const titles: Record<string, string> = {
  itemTimings: "Item purchase cutoffs", laneRoles: "Lane roles by maximum match duration", misc: "Team scenarios",
};
const lanes: Record<number, string> = { 1: "Safe lane", 2: "Mid lane", 3: "Off lane", 4: "Jungle" };

export default function Scenarios({ hero }: { hero: string }) {
  const [data, setData] = useState<Response | null>(null);
  const [error, setError] = useState("");
  const [limits, setLimits] = useState<Record<string, number>>({});
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch(`http://localhost:8000/scenarios?hero=${encodeURIComponent(hero)}`, { signal: controller.signal });
        if (!response.ok) throw new Error("Scenario outcomes unavailable. Retrying automatically.");
        const result: Response = await response.json();
        if (!controller.signal.aborted) { setData(result); setError(""); }
      } catch (err) {
        if (!controller.signal.aborted) setError(err instanceof Error ? err.message : "Scenario outcomes unavailable");
      }
    }
    void load();
    const timer = window.setInterval(() => void load(), 300000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [hero]);
  return <section aria-label="Scenario outcomes">
    <h2>Observed outcomes for {hero}</h2>
    <p>OpenDota aggregate statistics · cached for one hour · no Turbo filter</p>
    <p>Associations do not prove an item causes wins. Timing cutoffs overlap; do not add their game counts.</p>
    {error && <p role="status">{error}</p>}
    {data?.hero === hero ? Object.entries(data.scenarios).map(([kind, dataset]) => <details key={kind} open={kind === "itemTimings"}>
      <summary>{titles[kind]}{kind === "misc" && " (all heroes)"}</summary>
      {dataset.error ? <p role="status">{dataset.error}</p> : <>
        <p>Fetched: {dataset.fetched_at && new Date(dataset.fetched_at).toLocaleString()}{dataset.stale && " · Stale data: refresh unavailable"}</p>
        {dataset.rows?.length ? <><p>Showing {Math.min(limits[kind] ?? 20, dataset.rows.length)} of {dataset.rows.length} rows, largest samples first.</p><div className="outcome-table"><table>
          <thead><tr><th>Scenario / item</th><th>Time (minutes)</th><th>Games</th><th>Wins</th><th>Observed win rate</th></tr></thead>
          <tbody>{dataset.rows.slice(0, limits[kind] ?? 20).map((row, index) => <tr key={index}>
            <td>{row.item || (row.lane_role ? lanes[row.lane_role] : `${row.scenario} · ${row.is_radiant ? "Radiant" : "Dire"} · region ${row.region}`)}</td>
            <td>{row.time_minutes ?? "—"}</td><td>{row.games}</td><td>{row.wins}</td>
            <td>{row.win_rate_pct == null ? "No games" : `${row.win_rate_pct.toFixed(2)}%`}{row.small_sample && " · small sample (<100)"}</td>
          </tr>)}</tbody>
        </table></div>
          {(limits[kind] ?? 20) < dataset.rows.length && <button type="button"
            onClick={() => setLimits(current => ({ ...current, [kind]: (current[kind] ?? 20) + 20 }))}>Show 20 more rows</button>}
        </> : <p>No outcomes reported.</p>}
        <small>{dataset.limitation}</small>
      </>}
    </details>) : !error && <p role="status">Loading scenario outcomes…</p>}
  </section>;
}
