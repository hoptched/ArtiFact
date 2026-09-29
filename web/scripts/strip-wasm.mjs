// The ONNX runtime wasm is fetched from the CDN at run time, so the copy
// the bundler emits is dead weight: 25.6 MiB that nothing requests, and
// over the single-file limit some static hosts enforce.
import { readdir, rm, stat } from "node:fs/promises";
import { join } from "node:path";

const dir = "dist/assets";
let freed = 0;
for (const name of await readdir(dir)) {
  if (!name.endsWith(".wasm")) continue;
  const path = join(dir, name);
  freed += (await stat(path)).size;
  await rm(path);
  console.log(`  dropped ${name}`);
}
if (freed) console.log(`  ${(freed / 1048576).toFixed(1)} MiB not deployed`);
