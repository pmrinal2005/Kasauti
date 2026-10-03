/**
 * ts-loader.mjs — minimal Node loader so `node --test tests/*.test.ts` works on Node 20.
 *
 * Why this exists: Node's built-in TypeScript *type stripping* only arrived in 22.6, and this
 * project runs on Node 20 (the `engines` floor, and the version Kaggle/Vercel images ship). The
 * previous loader only rewrote extensionless `./x` specifiers and then handed the `.ts` file to
 * Node untouched, which fails with ERR_UNKNOWN_FILE_EXTENSION.
 *
 * Two hooks:
 *   resolve — `@/foo` → <repo>/src/foo(.ts|/index.ts), and extensionless relative specifiers get `.ts`.
 *   load    — `.ts`/`.tsx`/`.mts` is transpiled in memory with the `typescript` devDependency
 *             (already installed for `npm run typecheck`, so no new dependency).
 *
 * Type checking is NOT this loader's job — `npm run typecheck` does that against the real
 * tsconfig. This only has to produce runnable ESM.
 */
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve as resolvePath } from "node:path";
import ts from "typescript";

const ROOT = resolvePath(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = resolvePath(ROOT, "src");
const TS_RE = /\.(ts|tsx|mts)$/;
const CANDIDATES = (p) => [p, `${p}.ts`, `${p}.tsx`, `${p}/index.ts`, `${p}/index.tsx`];

function firstExisting(path) {
  for (const c of CANDIDATES(path)) if (existsSync(c)) return pathToFileURL(c).href;
  return null;
}

export async function resolve(specifier, context, next) {
  if (specifier.startsWith("@/")) {
    const hit = firstExisting(resolvePath(SRC, specifier.slice(2)));
    if (hit) return { url: hit, format: "module", shortCircuit: true };
  }
  if ((specifier.startsWith("./") || specifier.startsWith("../")) && !TS_RE.test(specifier) && !/\.[cm]?js$/.test(specifier)) {
    const parent = fileURLToPath(context.parentURL);
    const hit = firstExisting(resolvePath(dirname(parent), specifier));
    if (hit) return { url: hit, format: "module", shortCircuit: true };
  }
  return next(specifier, context);
}

export async function load(url, context, next) {
  if (!url.startsWith("file:") || !TS_RE.test(url)) return next(url, context);
  const source = readFileSync(fileURLToPath(url), "utf8");
  const out = ts.transpileModule(source, {
    fileName: fileURLToPath(url),
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
      verbatimModuleSyntax: false,
      isolatedModules: true,
      sourceMap: false,
    },
  });
  return { format: "module", source: out.outputText, shortCircuit: true };
}
