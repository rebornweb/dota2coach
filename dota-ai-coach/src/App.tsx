
import { useState } from "react";
import "./App.css";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

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
        body: JSON.stringify({ prompt: question })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data: { answer: string } = await response.json();

      setMessages(prev => [
        ...prev,
        { role: "assistant", content: data.answer }
      ]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Failed to connect to AI." }
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
        <input
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
