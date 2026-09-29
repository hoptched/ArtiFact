import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The ONNX runtime's wasm is 25.6 MiB, which is over what some static
// hosts will accept as a single file, and it is identical to the copy on
// the public CDN. Serving it from there keeps it out of the deployment
// and off our bandwidth. Read from the installed package rather than
// written down, so the URL cannot drift from the code that uses it.
// Read off disk rather than required: the package does not export its
// own manifest, so resolution refuses it.
const ortVersion: string = JSON.parse(
  readFileSync("node_modules/onnxruntime-web/package.json", "utf8")).version;

// base is relative so the built site works from a project subpath on
// GitHub Pages without knowing the repository name at build time.
export default defineConfig({
  plugins: [react()],
  base: "./",
  define: { __ORT_VERSION__: JSON.stringify(ortVersion) },
});
