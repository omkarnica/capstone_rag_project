import { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/landing.css";

export default function LandingPage() {
  const navigate = useNavigate();

  function enterWorkspace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    navigate("/workspace");
  }

  function continueAsGuest() {
    navigate("/workspace");
  }

  return (
    <main className="landing-shell">
      <section className="landing-panel" aria-label="Workspace access">
        <div className="landing-copy">
          <p className="landing-eyebrow">Retrieval-powered financial research</p>
          <h1>Smart Financial RAG</h1>
          <p className="landing-description">
            Sign in to explore filings, board relationships, and connected company context from
            one workspace.
          </p>
        </div>

        <form className="landing-form" onSubmit={enterWorkspace}>
          <label className="landing-field">
            <span>Email</span>
            <input type="email" name="email" autoComplete="email" placeholder="analyst@firm.com" />
          </label>

          <label className="landing-field">
            <span>Password</span>
            <input
              type="password"
              name="password"
              autoComplete="current-password"
              placeholder="Enter your password"
            />
          </label>

          <button className="landing-primary-action" type="submit">
            Enter Workspace
          </button>
          <button className="landing-secondary-action" type="button" onClick={continueAsGuest}>
            Continue as Guest
          </button>
        </form>
      </section>
    </main>
  );
}
