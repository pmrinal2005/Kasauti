/** SHA-256 (exact match) + 64-bit SimHash (near-duplicate match) fingerprints. */

export async function sha256Hex(text: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  const bytes = new Uint8Array(buf);
  let hex = "";
  for (let i = 0; i < bytes.length; i++) hex += bytes[i].toString(16).padStart(2, "0");
  return hex;
}

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK64 = 0xffffffffffffffffn;

function fnv1a64(s: string): bigint {
  let h = FNV_OFFSET;
  for (let i = 0; i < s.length; i++) {
    h ^= BigInt(s.charCodeAt(i));
    h = (h * FNV_PRIME) & MASK64;
  }
  return h;
}

/** Character 4-gram shingles → 64-bit SimHash, returned as 16-char hex. */
export function simhash64(text: string): string {
  const acc = new Int32Array(64);
  const t = text.replace(/\s+/g, " ");
  const n = 4;
  if (t.length < n) return "0".repeat(16);
  for (let i = 0; i <= t.length - n; i++) {
    const h = fnv1a64(t.slice(i, i + n));
    for (let b = 0; b < 64; b++) acc[b] += (h >> BigInt(b)) & 1n ? 1 : -1;
  }
  let out = 0n;
  for (let b = 0; b < 64; b++) if (acc[b] > 0) out |= 1n << BigInt(b);
  return out.toString(16).padStart(16, "0");
}

export function hamming(a: string, b: string): number {
  let x = BigInt("0x" + a) ^ BigInt("0x" + b);
  let c = 0;
  while (x) {
    x &= x - 1n;
    c++;
  }
  return c;
}
