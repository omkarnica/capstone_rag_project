import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import App from "../App";
import { renderWithRouter } from "../test/renderWithRouter";

describe("App routing", () => {
  it("renders the landing page and enters the workspace", async () => {
    const { user } = renderWithRouter(<App />, { route: "/" });

    expect(screen.getByRole("heading", { name: /smart financial rag/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /enter workspace/i })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /enter workspace/i }));

    expect(screen.getByRole("heading", { name: /ask a question/i })).toBeInTheDocument();
  });
});
