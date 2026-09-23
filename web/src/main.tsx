import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

// Page zoom fights the map's zoom: both answer ctrl+scroll and trackpad
// pinch, and the browser's version resizes the interface out from under
// the gesture. Suppressed document-wide so it cannot happen off-canvas
// either. Keyboard zoom is deliberately left alone — it is how people
// with low vision read a page, and it does not collide with anything.
document.addEventListener(
  "wheel",
  (e) => { if (e.ctrlKey) e.preventDefault(); },
  { passive: false },
);
for (const type of ["gesturestart", "gesturechange", "gestureend"]) {
  document.addEventListener(type, (e) => e.preventDefault());
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
