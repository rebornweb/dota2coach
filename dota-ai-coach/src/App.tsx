
import { useState } from "react";
import "./App.css";

type Message = {
  role: "user" | "assistant";
  content: string;
  failed?: boolean;
};

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [hp, setHp] = useState("");
  const [enemyBurst, setEnemyBurst] = useState("");

  async function sendMessage(e: React.FormEvent<HTMLFormElement>) {
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
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: question,
          hp: hp === "" ? null : Number(hp),
          enemy_burst: enemyBurst === "" ? null : Number(enemyBurst),
          history: messages.filter(message => !message.failed).slice(-40)
            .map(({ role, content }) => ({ role, content })),
        })
      });

      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(typeof error?.detail === "string"
          ? error.detail : `Request failed (HTTP ${response.status}).`);
      }

      const data: {
        ml_prediction: "Defense" | "Damage" | null;
        answer: string;
      } = await response.json();

      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: data.ml_prediction
            ? `Experimental prediction: ${data.ml_prediction}\n\n${data.answer}`
            : data.answer,
        }
      ]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: error instanceof Error ? error.message : "Failed to connect to AI.",
          failed: true,
        }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>Dota 2 AI Coach</h1>
      <p>Powered by Ollama — running locally</p>

      <section className="chat">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <strong>{message.role === "user" ? "You" : "AI"}</strong>
            <p>{message.content}</p>
          </div>
        ))}
        {loading && <p>AI is thinking...</p>}
      </section>

      <form onSubmit={sendMessage}>
        <fieldset className="game-context" disabled={loading}>
          <legend>Optional prediction context: Dazzle in Turbo</legend>
          <label>
            HP (%)
            <input
              type="number"
              min="0"
              max="100"
              step="any"
              value={hp}
              onChange={e => setHp(e.target.value)}
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
            />
          </label>
        </fieldset>
        <input
          aria-label="Strategy question"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Ask about Dota strategy..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !prompt.trim()}>
          Send
        </button>
      </form>
    </main>
  );
}
