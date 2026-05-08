import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import App from "../App";
import { renderWithRouter } from "../test/renderWithRouter";

describe("EvaluationPage", () => {
  it("renders the evaluation workspace with run controls and status cards", () => {
    vi.restoreAllMocks();

    renderWithRouter(<App />, { route: "/evaluation" });

    expect(screen.getByRole("heading", { name: /evaluation/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run evaluation/i })).toBeInTheDocument();
    expect(screen.getByText(/last run/i)).toBeInTheDocument();
    expect(screen.getByText(/configs tested/i)).toBeInTheDocument();
    expect(screen.getByText(/dataset size/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /workspace/i })).toBeInTheDocument();
  });
});
