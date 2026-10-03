// Test-only loader: lets Node's built-in type stripping resolve extensionless `./x` imports in src/core.
import { register } from "node:module";
register("data:text/javascript," + encodeURIComponent(`
export async function resolve(spec, ctx, next) {
  try { return await next(spec, ctx); }
  catch (e) {
    if ((spec.startsWith("./") || spec.startsWith("../")) && !/\\.[cm]?[jt]s$/.test(spec)) return next(spec + ".ts", ctx);
    throw e;
  }
}`));
