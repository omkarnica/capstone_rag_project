import { ReactElement } from "react";
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

export function renderWithRouter(ui: ReactElement, options?: { route?: string }) {
  return {
    user: userEvent.setup(),
    ...render(
      <MemoryRouter initialEntries={[options?.route ?? "/"]}>
        {ui}
      </MemoryRouter>,
    ),
  };
}
