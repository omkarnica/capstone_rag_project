import { Link, useLocation } from "react-router-dom";

const TABS = [
  { label: "Workspace", to: "/workspace" },
  { label: "Evaluation", to: "/evaluation" },
];

export default function ProjectTabs() {
  const location = useLocation();

  return (
    <nav className="project-tabs" aria-label="Project sections">
      {TABS.map((tab) => {
        const active = location.pathname === tab.to;
        return (
          <Link key={tab.to} to={tab.to} className={active ? "project-tab is-active" : "project-tab"}>
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
