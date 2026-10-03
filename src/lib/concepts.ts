/**
 * Static, human-reviewed concept scripts (Voice-First Explainer).
 * Never names a specific fund, stock or broker. Spoken via on-device SpeechSynthesis.
 */
export interface Concept {
  id: string;
  title: string;
  family: "basics" | "costs" | "risk" | "red_flags" | "safety";
  /**
   * Routing keywords for the Tier-0 lexical router. Coverage here is deliberately conservative —
   * a wrong direct hit is worse than a near miss that shows the disambiguation cards — and the
   * list is expected to grow from real transcripts, not from guesses. When a calibrated
   * `intent_concepts_v1` export is loaded, Laya refines this routing (see views/Voice.tsx).
   */
  keys: string[];
  text: Record<"en-IN" | "hi-IN", string>;
}

export const CONCEPTS: Concept[] = [
  { id: "nav", title: "NAV — Net Asset Value", family: "basics", keys: ["nav", "net asset", "unit price", "एनएवी"],
    text: { "en-IN": "Think of a thali. NAV is the price of one plate, worked out by adding up everything on the table and dividing by the number of plates. A low NAV is not 'cheap' and a high NAV is not 'expensive' — it only tells you the price of one unit today.", "hi-IN": "एक थाली सोचिए। NAV एक थाली की कीमत है — सारी चीज़ों का कुल मूल्य, थालियों की संख्या से भाग। कम NAV सस्ता नहीं और ज़्यादा NAV महंगा नहीं होता — यह सिर्फ आज एक यूनिट का दाम है।" } },
  { id: "sip", title: "SIP — Systematic Investment Plan", family: "basics", keys: ["sip", "monthly invest", "systematic", "एसआईपी", "sip kya"],
    text: { "en-IN": "A SIP is like filling a matka one glass at a time. You invest a fixed amount every month. When prices are low you get more units, when they are high you get fewer. It builds discipline — but it does not guarantee profit.", "hi-IN": "SIP ऐसा है जैसे मटका रोज़ एक गिलास से भरना। हर महीने तय रकम लगती है। दाम कम हों तो ज़्यादा यूनिट, ज़्यादा हों तो कम। इससे अनुशासन बनता है — पर मुनाफ़े की गारंटी नहीं।" } },
  { id: "diversification", title: "Diversification", family: "risk", keys: ["diversif", "spread", "eggs", "basket", "विविध"],
    text: { "en-IN": "A wise farmer never plants only one crop. If the rain fails one, the others still feed the family. Spreading money across different kinds of investments works the same way — it lowers the damage of any single loss.", "hi-IN": "समझदार किसान एक ही फ़सल नहीं बोता। बारिश एक को बिगाड़े तो बाकी परिवार का पेट भरती हैं। पैसा अलग-अलग जगह लगाना भी ऐसा ही है।" } },
  { id: "volatility", title: "Volatility", family: "risk", keys: ["volatil", "ups and downs", "fluctuat", "उतार चढ़ाव"],
    text: { "en-IN": "Markets move like the monsoon — some weeks heavy rain, some weeks dry. Short-term ups and downs are normal. Anyone who promises only sunshine is not describing a real market.", "hi-IN": "बाज़ार मानसून जैसा चलता है — कभी तेज़ बारिश, कभी सूखा। छोटे समय के उतार-चढ़ाव आम हैं। जो सिर्फ धूप का वादा करे, वह असली बाज़ार की बात नहीं कर रहा।" } },
  { id: "compounding", title: "Compounding", family: "basics", keys: ["compound", "interest on interest", "चक्रवृद्धि"],
    text: { "en-IN": "A mango tree gives fruit, the seeds grow new trees, and those trees give more fruit. Returns that earn their own returns grow slowly at first and faster over many years. Time matters more than timing.", "hi-IN": "आम का पेड़ फल देता है, उसकी गुठली से नए पेड़ उगते हैं, जो और फल देते हैं। मुनाफ़े पर मुनाफ़ा पहले धीरे, फिर सालों में तेज़ बढ़ता है।" } },
  { id: "expense_ratio", title: "Expense ratio", family: "costs", keys: ["expense ratio", "fee", "charges", "commission", "खर्च"],
    text: { "en-IN": "Like the mandi commission agent who keeps a small cut of every sale, a fund keeps a yearly percentage as its fee. One percent sounds small, but over twenty years it can eat a large slice of your growth.", "hi-IN": "जैसे मंडी का आढ़ती हर बिक्री में थोड़ा हिस्सा रखता है, वैसे ही फंड हर साल एक प्रतिशत फ़ीस रखता है। एक प्रतिशत छोटा लगता है, पर बीस साल में बड़ा हिस्सा खा जाता है।" } },
  { id: "nomination", title: "Nomination", family: "safety", keys: ["nomination", "nominee", "नामांकन", "नॉमिनी"],
    text: { "en-IN": "Nomination is like leaving a spare house key with someone you trust. If something happens to you, your family can reach your investments without long paperwork. Add a nominee to every account.", "hi-IN": "नॉमिनेशन ऐसा है जैसे भरोसेमंद व्यक्ति के पास घर की दूसरी चाबी रखना। हर खाते में नॉमिनी ज़रूर जोड़ें।" } },
  { id: "guaranteed_return", title: "“Guaranteed return” — a red flag", family: "red_flags", keys: ["guarantee", "guaranteed", "assured", "risk free", "गारंटी", "पक्का"],
    text: { "en-IN": "No market investment registered with SEBI can promise a guaranteed return. If someone says 'guaranteed', 'fixed daily profit' or 'zero risk', treat it as a warning sign and verify the person on SEBI's website.", "hi-IN": "SEBI में रजिस्टर्ड कोई भी बाज़ार निवेश गारंटीड रिटर्न का वादा नहीं कर सकता। 'गारंटी', 'रोज़ का पक्का मुनाफ़ा' या 'ज़ीरो रिस्क' सुनें तो सावधान रहें।" } },
  { id: "vip_group", title: "“VIP tips group” — a red flag", family: "red_flags", keys: ["vip", "group", "telegram", "whatsapp group", "tips", "ग्रुप"],
    text: { "en-IN": "The most common investment scam starts with an invitation to a VIP WhatsApp or Telegram group that shares 'winning tips', then moves you to a fake trading app that shows fake profits. Real advisers are registered and never ask you to pay into personal UPI IDs.", "hi-IN": "सबसे आम निवेश ठगी VIP व्हाट्सऐप या टेलीग्राम ग्रुप के न्योते से शुरू होती है, फिर नकली ट्रेडिंग ऐप पर नकली मुनाफ़ा दिखाया जाता है। असली सलाहकार रजिस्टर्ड होते हैं।" } },
  // The safest script for "someone defrauded me, what now?" is this one — it is the only concept that
  // ends with the 1930 / cybercrime.gov.in path — so the reporting vocabulary lives here too.
  { id: "digital_arrest", title: "“Digital arrest” — always a scam", family: "safety", keys: ["digital arrest", "cbi", "police", "arrest", "customs", "अरेस्ट", "report fraud", "report a fraud", "fraud report", "cybercrime", "1930", "helpline", "money back", "ठगी", "धोखाधड़ी"],
    text: { "en-IN": "There is no such thing as a digital arrest. Real police, CBI, customs or SEBI officials never keep you on a video call or ask you to transfer money to clear your name. Hang up, call your trusted contact, and call 1930.", "hi-IN": "डिजिटल अरेस्ट जैसी कोई चीज़ नहीं होती। असली पुलिस या CBI कभी वीडियो कॉल पर रोककर पैसे नहीं मंगवाती। फ़ोन काटें, अपने भरोसेमंद व्यक्ति को बताएं, और 1930 पर कॉल करें।" } },
];

export function routeConcept(transcript: string): { concept: Concept | null; alternatives: Concept[]; score: number } {
  const t = transcript.toLowerCase();
  const scored = CONCEPTS.map((c) => ({ c, s: c.keys.reduce((s, k) => s + (t.includes(k) ? k.length : 0), 0) })).sort((a, b) => b.s - a.s);
  const top = scored[0];
  if (!top || top.s === 0) return { concept: null, alternatives: scored.slice(0, 3).map((x) => x.c), score: 0 };
  const margin = top.s - (scored[1]?.s ?? 0);
  // abstain when two concepts tie — show disambiguation cards instead of guessing
  if (margin === 0) return { concept: null, alternatives: scored.slice(0, 3).map((x) => x.c), score: top.s };
  return { concept: top.c, alternatives: scored.slice(1, 4).map((x) => x.c), score: top.s };
}

/** Lookup by concept id — the concept library is the addressable set of spoken scripts. */
export const CONCEPT_BY_ID: Record<string, Concept> = Object.fromEntries(CONCEPTS.map((c) => [c.id, c]));

/**
 * The Laya `intent_concepts_v1` bank and this library grew separately, so their keys differ in two
 * places. The bank is the authority for *routing*; this map is the authority for *what we can say*.
 * A bank key with no script (exit_load) resolves to null, and the caller offers the family instead
 * of inventing an answer.
 */
export const BANK_KEY_TO_CONCEPT: Record<string, string> = {
  guaranteed: "guaranteed_return",
  reporting: "digital_arrest",
  fraud_report: "digital_arrest",
  vip_tips: "vip_group",
};

/** When Laya names a family but no leaf concept, the family's safest script is offered instead. */
export const FAMILY_FALLBACK: Record<string, string> = {
  basics: "nav",
  costs: "expense_ratio",
  risk: "diversification",
  red_flags: "guaranteed_return",
  safety: "nomination",
};

export function conceptForBankKey(key: string | undefined): Concept | null {
  if (!key) return null;
  return CONCEPT_BY_ID[key] ?? CONCEPT_BY_ID[BANK_KEY_TO_CONCEPT[key] ?? ""] ?? null;
}

/** Resolve a bank family (triage choice) to something we can actually speak. */
export function conceptForFamily(family: string | undefined): Concept | null {
  if (!family) return null;
  return CONCEPT_BY_ID[family] ?? conceptForBankKey(FAMILY_FALLBACK[family]);
}
