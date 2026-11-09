
import { useState, type FormEvent } from "react";
import "./App.css";

type Message = {
  role: "user" | "assistant";
  content: string;
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

const HEROES = [
  "Dazzle",
  "Earthshaker",
  "Anti-Mage",
  "Phantom Lancer",
  "Axe",
  "Ogre Magi",
  "Pudge",
  "Sniper",
  "Witch Doctor",
  "Viper",
  "Luna",
  "Legion Commander",
];

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  // Game context
  const [hero, setHero] = useState("Dazzle");
  const [gameMode, setGameMode] = useState("Turbo");
  const [hp, setHp] = useState("");
  const [enemyBurst, setEnemyBurst] = useState("");

  async function sendMessage(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();

    if (!prompt.trim() || loading) return;

    const question = prompt.trim();

    setPrompt("");
    setLoading(true);

    setMessages(prev => [
      ...prev,
      { role: "user", content: question }
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
          content
        }
      ]);

    } catch (error) {
      console.error(error);

      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
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
              {HEROES.map(name => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
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
      
      <section className="chat">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role}`}
          >
            <strong>
              {message.role === "user" ? "You" : "AI"}
            </strong>

            <p>{message.content}</p>
          </div>
        ))}

        {loading && <p>AI is thinking...</p>}
      </section>

    </main>
  );
}
