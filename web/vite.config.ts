import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base is relative so the built site works from a project subpath on
// GitHub Pages without knowing the repository name at build time.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
