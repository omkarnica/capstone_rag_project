import { Navigate, Route, Routes } from "react-router-dom";
import EvaluationPage from "./routes/EvaluationPage";
import LandingPage from "./routes/LandingPage";
import WorkspacePage from "./routes/WorkspacePage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/workspace" element={<WorkspacePage />} />
      <Route path="/evaluation" element={<EvaluationPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
