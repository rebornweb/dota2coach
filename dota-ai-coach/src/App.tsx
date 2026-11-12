
import { useEffect, useState, type FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

type Message = {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  failed?: boolean;
};

type HeroPrediction = {
  hero?: string;
  estimated_turbo_win_rate?: number;
  observed_turbo_win_rate?: number | null;
  limitation?: string;
  error?: string;
};

type ChatResponse = {
  answer: string;
  ml_prediction: "Defense" | "Damage" | null;
  hero_prediction?: HeroPrediction | null;
};

type ItemPopularity = {
  hero: string;
  fetched_at: string;
  stale: boolean;
  items_stale: boolean;
  limitation: string;
  stages: Record<string, { item_id: number; name: string; count: number; purchase_share_pct: number }[]>;
};

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  // Game context
  const [hero, setHero] = useState("Dazzle");
  const [gameMode, setGameMode] = useState("Turbo");
  const [hp, setHp] = useState("");
  const [enemyBurst, setEnemyBurst] = useState("");
  const [heroes, setHeroes] = useState<string[]>(["Dazzle"]);
  const [heroListError, setHeroListError] = useState("");
  const [itemData, setItemData] = useState<ItemPopularity | null>(null);
  const [itemError, setItemError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    async function loadItems() {
      try {
        const response = await fetch(`http://localhost:8000/item-popularity?hero=${encodeURIComponent(hero)}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Item popularity unavailable. Retrying automatically.");
        const data: ItemPopularity = await response.json();
        if (!controller.signal.aborted) { setItemData(data); setItemError(""); }
      } catch (error) {
        if (!controller.signal.aborted) setItemError(error instanceof Error ? error.message : "Item data unavailable");
      }
    }
    void loadItems();
    const timer = window.setInterval(() => void loadItems(), 60000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [hero]);

  useEffect(() => {
    const controller = new AbortController();
    async function loadHeroes() {
      try {
        const response = await fetch("http://localhost:8000/heroes", {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const names: unknown = await response.json();
        if (!Array.isArray(names) || names.length === 0
            || !names.every(name => typeof name === "string" && name.trim())) {
          throw new Error("Invalid hero list");
        }
        setHeroes(names);
        setHero(current => names.includes(current) ? current : names[0]);
      } catch {
        if (!controller.signal.aborted) {
          setHeroListError("Could not load heroes. Dazzle is available; reload to retry.");
        }
      }
    }
    void loadHeroes();
    return () => controller.abort();
  }, []);

  async function sendMessage(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();

    if (!prompt.trim() || loading) return;

    const question = prompt.trim();

    setPrompt("");
    setLoading(true);

    setMessages(prev => [
      ...prev,
      { role: "user", content: question, timestamp: new Date().toISOString() }
    ]);

    try {
      const response = await fetch(
        "http://localhost:8000/chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            prompt: question,

            // Selected game context
            hero: hero,
            game_mode: gameMode,

            // Item ML inputs are currently Dazzle Turbo only
            hp:
              hero === "Dazzle" && gameMode === "Turbo" && hp !== ""
                ? Number(hp)
                : null,

            enemy_burst:
              hero === "Dazzle" && gameMode === "Turbo" && enemyBurst !== ""
                ? Number(enemyBurst)
                : null,

            // Existing conversation history
            history: messages
              .filter(message => !message.failed)
              .slice(-40)
              .map(({ role, content }) => ({
                role,
                content
              }))
          })
        }
      );

      if (!response.ok) {
        const error = await response.json().catch(() => null);

        throw new Error(
          typeof error?.detail === "string"
            ? error.detail
            : `Request failed (HTTP ${response.status}).`
        );
      }

      const data: ChatResponse = await response.json();

      const details: string[] = [];

      // Model 1: Item priority classifier
      if (data.ml_prediction) {
        details.push(
          `Experimental item prediction: ${data.ml_prediction}`
        );
      }

      // Model 2: Hero performance regression
      if (data.hero_prediction) {
        const stats = data.hero_prediction;

        if (stats.error) {
          details.push(`Hero data: ${stats.error}`);
        } else {
          if (stats.estimated_turbo_win_rate != null) {
            details.push(
              `Estimated Turbo win rate: ${stats.estimated_turbo_win_rate.toFixed(2)}%`
            );
          }

          if (stats.observed_turbo_win_rate != null) {
            details.push(
              `Observed Turbo win rate: ${stats.observed_turbo_win_rate.toFixed(2)}%`
            );
          }
          if (stats.limitation) details.push(stats.limitation);
        }
      }

      const content = [
        ...details,
        data.answer
      ].join("\n\n");

      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content,
          timestamp: new Date().toISOString()
        }
      ]);

    } catch (error) {
      console.error(error);

      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          timestamp: new Date().toISOString(),
          content:
            error instanceof Error
              ? error.message
              : "Failed to connect to AI.",
          failed: true
        }
      ]);

    } finally {
      setLoading(false);
    }
  }

  const itemPredictionAvailable =
    hero === "Dazzle" && gameMode === "Turbo";

  return (
    <main className="container">
      <h1>Dota 2 AI Coach</h1>
      <p>Powered by Ollama — running locally</p>

      <form onSubmit={sendMessage}>

        <fieldset
          className="game-context"
          disabled={loading}
        >
          <legend>Game Context</legend>

          <label>
            Hero
            <select
              value={hero}
              onChange={e => setHero(e.target.value)}
            >
              {heroes.map(name => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            {heroListError && <small role="status">{heroListError}</small>}
          </label>

          <label>
            Game Mode
            <select
              value={gameMode}
              onChange={e => setGameMode(e.target.value)}
            >
              <option value="Turbo">Turbo</option>
              <option value="All Pick">All Pick</option>
            </select>
          </label>

          <label>
            HP (%)
            <input
              type="number"
              min="0"
              max="100"
              step="any"
              value={hp}
              onChange={e => setHp(e.target.value)}
              disabled={!itemPredictionAvailable}
              placeholder="0–100"
            />
          </label>

          <label>
            Enemy burst threat (0–10)
            <input
              type="number"
              min="0"
              max="10"
              step="any"
              value={enemyBurst}
              onChange={e => setEnemyBurst(e.target.value)}
              disabled={!itemPredictionAvailable}
              placeholder="0–10"
            />
          </label>

          {!itemPredictionAvailable && (
            <small>
              Item priority ML is currently available
              only for Dazzle in Turbo.
            </small>
          )}
        </fieldset>
        <label className="strategy-question">
          Strategy question
        <input
          aria-label="Strategy question"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder={`Ask about ${hero} in ${gameMode}...`}
          disabled={loading}
        />
        </label>
        <button
          type="submit"
          disabled={loading || !prompt.trim()}
        >
          {loading ? "Thinking..." : "Send"}
        </button>

        <button
          type="button"
          disabled={loading || messages.length === 0}
          onClick={() => setMessages([])}
        >
          Clear chat
        </button>

      </form>
      
      <section aria-label="Item popularity">
        <h2>Popular items for {hero}</h2>
        <p>OpenDota aggregate purchases · refreshed every 5 minutes · no Turbo filter</p>
        {itemError && <p role="status">{itemError}</p>}
        {itemData?.hero === hero ? <>
          <p>Fetched: {new Date(itemData.fetched_at).toLocaleString()}
            {itemData.stale && " · Stale data: OpenDota unavailable"}
            {itemData.items_stale && " · Item names unavailable or stale"}</p>
          {Object.entries(itemData.stages).map(([stage, rows]) => <details key={stage}>
            <summary>{stage.replaceAll("_", " ")}</summary>
            {rows.length ? <ol>{rows.slice(0, 10).map(item => <li key={item.item_id}>
              {item.name}: {item.count} purchases ({item.purchase_share_pct}% of stage purchases)
            </li>)}</ol> : <p>No purchases reported.</p>}
          </details>)}
          <small>{itemData.limitation}</small>
        </> : !itemError && <p role="status">Loading item popularity…</p>}
      </section>
      <section className="chat" aria-label="Conversation">
        {loading && <p className="chat-status" role="status">AI Coach is thinking...</p>}
        {messages.slice().reverse().map((message, index) => (
          <div
            key={messages.length - 1 - index}
            className={`message ${message.role}`}
          >
            <strong className="message-label">
              <span className="message-avatar" aria-hidden="true">
                {message.role === "user" ? "Y" : "AI"}
              </span>
              {message.role === "user" ? "You" : "AI Coach"}
              <time
                className="message-time"
                dateTime={message.timestamp}
                title={new Date(message.timestamp).toLocaleString()}
              >
                {new Date(message.timestamp).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </time>
            </strong>

            <div className="message-content">
            {message.role === "assistant" ? (
              <ReactMarkdown>{message.content}</ReactMarkdown>
            ) : (
              <p>{message.content}</p>
            )}
            </div>
          </div>
        ))}

      </section>

    </main>
  );
}
