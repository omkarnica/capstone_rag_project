import { beforeEach, describe, expect, it, vi } from "vitest";
import { clearCache, runQuery } from "./api";

describe("api client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("posts query requests to the backend service url", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ final_answer: "ok" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await runQuery({ question: "What was Apple's revenue?" });

    expect(fetchMock).toHaveBeenCalledWith(
      "https://ma-oracle-508519534978.us-central1.run.app/query",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("posts cache clear requests to the backend service url", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ status: "ok", cleared: {} }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await clearCache();

    expect(fetchMock).toHaveBeenCalledWith(
      "https://ma-oracle-508519534978.us-central1.run.app/cache/clear",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
