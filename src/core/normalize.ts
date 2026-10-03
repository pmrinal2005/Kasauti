/**
 * Text normalisation — framework-free, allocation-light, runs inside a Worker.
 * Defeats the common obfuscation tricks used in forwarded scam messages:
 * zero-width joiners, Cyrillic/Greek homoglyphs, native-script digits,
 * full-width forms and tracking-param-laden URLs.
 */

const ZERO_WIDTH = /[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u00AD\u180E]/g;

// Latin look-alikes from Cyrillic / Greek / misc blocks → ASCII.
const HOMOGLYPHS: Record<string, string> = {
  а: "a", е: "e", о: "o", р: "p", с: "c", у: "y", х: "x", і: "i", ј: "j", ѕ: "s", һ: "h", ԁ: "d", ӏ: "l", ԛ: "q", ԝ: "w",
  А: "a", В: "b", Е: "e", К: "k", М: "m", Н: "h", О: "o", Р: "p", С: "c", Т: "t", Х: "x", У: "y", І: "i",
  α: "a", ο: "o", ρ: "p", ε: "e", ι: "i", κ: "k", ν: "v", τ: "t", υ: "u", χ: "x", Α: "a", Β: "b", Ε: "e", Ζ: "z",
  Η: "h", Ι: "i", Κ: "k", Μ: "m", Ν: "n", Ο: "o", Ρ: "p", Τ: "t", Χ: "x", Υ: "y",
  "\u0131": "i", "\u0261": "g", "\u01C3": "!",
};

// Zero-digit code points of Indic + Arabic-Indic scripts. digit = cp - zero.
const DIGIT_ZEROS = [0x0966, 0x09e6, 0x0a66, 0x0ae6, 0x0b66, 0x0be6, 0x0c66, 0x0ce6, 0x0d66, 0x0660, 0x06f0];

function foldChar(ch: string): string {
  const cp = ch.codePointAt(0)!;
  for (let i = 0; i < DIGIT_ZEROS.length; i++) {
    const z = DIGIT_ZEROS[i];
    if (cp >= z && cp <= z + 9) return String.fromCharCode(48 + cp - z);
  }
  return HOMOGLYPHS[ch] ?? ch;
}

const TRACKING = /^(utm_|fbclid|gclid|igshid|si$|ref$|mc_)/i;

export function canonicalUrl(raw: string): string {
  try {
    const u = new URL(raw.startsWith("http") ? raw : `https://${raw}`);
    u.hostname = u.hostname.toLowerCase().replace(/^www\./, "");
    u.hash = "";
    for (const k of [...u.searchParams.keys()]) if (TRACKING.test(k)) u.searchParams.delete(k);
    return u.toString().replace(/\/$/, "");
  } catch {
    return raw.toLowerCase();
  }
}

export interface NormalizeResult {
  text: string;
  /** count of obfuscation characters removed/folded — itself a weak signal */
  obfuscation: number;
}

export function normalize(input: string): NormalizeResult {
  let obfuscation = 0;
  const nfkc = input.normalize("NFKC");
  const stripped = nfkc.replace(ZERO_WIDTH, () => {
    obfuscation++;
    return "";
  });
  let out = "";
  for (const ch of stripped) {
    const f = foldChar(ch);
    if (f !== ch && !/[0-9]/.test(f)) obfuscation++;
    out += f;
  }
  out = out
    .toLowerCase()
    .replace(/https?:\/\/[^\s]+/g, (m) => canonicalUrl(m))
    .replace(/\s+/g, " ")
    .trim();
  return { text: out, obfuscation };
}
