import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import App from "../App";
import * as api from "../lib/api";
import { renderWithRouter } from "../test/renderWithRouter";

vi.mock("../lib/api");

const mockedRunQuery = vi.mocked(api.runQuery);
const mockedClearCache = vi.mocked(api.clearCache);

describe("WorkspacePage", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockedRunQuery.mockResolvedValue({
      final_answer: "Apple reported strong revenue.",
      final_route: "sql",
      cache_status: "miss",
      response_source: "retrieval_pipeline",
      citations: ["XBRL/SQL"],
      retrieved_contexts: ["[Document 1] XBRL/SQL\nApple revenue context"],
      retrieved_context_count: 1,
      plan_type: "single",
    });
    mockedClearCache.mockResolvedValue({ status: "ok", cleared: { exact: 1 } });
  });

  it("renders the workspace shell with key sections", () => {
    renderWithRouter(<App />, { route: "/workspace" });

    expect(screen.getByRole("heading", { name: /ask a question/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run query/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /clear cache/i })).toBeInTheDocument();
    expect(screen.getByText(/final answer/i)).toBeInTheDocument();
  });

  it("runs a query and renders backend result metadata", async () => {
    const { user } = renderWithRouter(<App />, { route: "/workspace" });
    const nowSpy = vi.spyOn(performance, "now");
    nowSpy.mockReturnValueOnce(1000);
    nowSpy.mockReturnValueOnce(1123);

    await user.type(screen.getByPlaceholderText(/type your query here/i), "What was Apple's revenue?");
    await user.click(screen.getByRole("button", { name: /run query/i }));

    await screen.findByText(/apple reported strong revenue\./i);

    expect(mockedRunQuery).toHaveBeenCalledWith({ question: "What was Apple's revenue?" });
    expect(screen.getByText("sql", { selector: "dd" })).toBeInTheDocument();
    expect(screen.getByText("miss", { selector: "dd" })).toBeInTheDocument();
    expect(screen.getByText(/ms$/, { selector: "dd" })).toBeInTheDocument();
    expect(screen.getByText("XBRL/SQL", { selector: "li" })).toBeInTheDocument();

    nowSpy.mockRestore();
  });

  it("shows an inline error if the query fails", async () => {
    mockedRunQuery.mockRejectedValueOnce(new Error("backend unavailable"));
    const { user } = renderWithRouter(<App />, { route: "/workspace" });

    await user.type(screen.getByPlaceholderText(/type your query here/i), "What was Apple's revenue?");
    await user.click(screen.getByRole("button", { name: /run query/i }));

    expect(await screen.findByText(/backend unavailable/i)).toBeInTheDocument();
  });

  it("clears cache through the backend endpoint", async () => {
    const { user } = renderWithRouter(<App />, { route: "/workspace" });

    await user.click(screen.getByRole("button", { name: /clear cache/i }));

    await waitFor(() => {
      expect(mockedClearCache).toHaveBeenCalled();
    });
  });
});
