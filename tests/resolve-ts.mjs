// Test-only loader registration. See tests/ts-loader.mjs for the resolve/load hooks.
// Used as `node --import ./tests/resolve-ts.mjs --test tests/*.test.ts`.
import { register } from "node:module";
register("./ts-loader.mjs", import.meta.url);
