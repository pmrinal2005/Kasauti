/**
 * Tier-0 scam lexicon. Multilingual (English, romanised Hindi/Hinglish,
 * Devanagari Hindi & Marathi, Tamil). Patterns are matched AFTER normalisation
 * (lower-cased, homoglyph-folded), so entries are lower-case.
 *
 * Each family maps to a published I4C / SEBI / RBI advisory pattern.
 */

export type FlagFamily =
  | "guaranteed_return"
  | "urgency"
  | "pay_to_withdraw"
  | "vip_group"
  | "impersonation"
  | "secrecy"
  | "remote_access"
  | "kyc_threat"
  | "insider_tip"
  | "defensive";

export interface FamilyMeta {
  label: string;
  weight: number; // contribution to the fused risk score
  advice: string;
}

export const FAMILY_META: Record<FlagFamily, FamilyMeta> = {
  guaranteed_return: { label: "Guaranteed return", weight: 3, advice: "No SEBI-registered product can guarantee market returns." },
  urgency: { label: "Artificial urgency", weight: 1.5, advice: "Pressure to act 'today only' is a classic manipulation lever." },
  pay_to_withdraw: { label: "Pay-to-withdraw", weight: 4, advice: "Being asked to pay 'tax' or 'fees' to release profits is the signature of investment fraud." },
  vip_group: { label: "VIP / tips group", weight: 2, advice: "Unregistered 'VIP' WhatsApp/Telegram groups are the main entry point of the I4C-documented script." },
  impersonation: { label: "Authority impersonation", weight: 4, advice: "Police, CBI, ED, customs or SEBI never conduct 'digital arrest' or demand money on video calls." },
  secrecy: { label: "Secrecy demand", weight: 2.5, advice: "Being told not to tell family is a deliberate isolation tactic." },
  remote_access: { label: "Remote-access app", weight: 3, advice: "Never install AnyDesk/TeamViewer/QuickSupport at a stranger's request." },
  kyc_threat: { label: "KYC / account-block threat", weight: 2.5, advice: "Banks do not block accounts via SMS links; open your bank app directly." },
  insider_tip: { label: "Insider / sure-shot tip", weight: 2, advice: "'Operator-driven' or 'insider' tips are illegal and usually pump-and-dump." },
  defensive: { label: "Awareness / defensive context", weight: -2.5, advice: "Text appears to warn about scams rather than promote one." },
};

export const LEXICON: Record<FlagFamily, string[]> = {
  guaranteed_return: [
    "guaranteed return", "guaranteed profit", "assured return", "fixed return", "risk free", "risk-free", "zero risk", "no risk",
    "double your money", "money double", "paisa double", "guaranteed income", "100% profit", "sure profit", "pakka profit",
    "guaranteed milega", "return milega", "गारंटीड", "गारंटी रिटर्न", "पैसा डबल", "निश्चित रिटर्न", "हमी परतावा", "खात्रीशीर परतावा",
    "உறுதியான லாபம்", "இரட்டிப்பு", "daily profit", "daily income", "per day profit",
  ],
  urgency: [
    "today only", "last chance", "limited seats", "only few slots", "hurry", "act now", "immediately", "within 24 hours",
    "offer ends", "jaldi karo", "abhi join", "turant", "तुरंत", "आज ही", "जल्दी", "लगेच", "உடனே", "இன்றே",
  ],
  pay_to_withdraw: [
    "pay tax to withdraw", "withdrawal fee", "processing fee to withdraw", "unlock your profit", "release your funds",
    "pay to withdraw", "withdrawal tax", "security deposit to withdraw", "upgrade your account to withdraw", "clearance fee",
    "income tax charge before withdrawal", "withdraw karne ke liye", "निकालने के लिए शुल्क", "निकासी शुल्क", "पैसे काढण्यासाठी",
    "திரும்பப் பெற கட்டணம்",
  ],
  vip_group: [
    "vip group", "vip channel", "premium group", "telegram group", "whatsapp group", "stock tips group", "trading group",
    "join our group", "institutional account", "ipo allotment guaranteed", "block deal", "otc trading", "vip ग्रुप", "टिप्स ग्रुप",
  ],
  impersonation: [
    "digital arrest", "cbi officer", "ed officer", "enforcement directorate", "customs department", "narcotics", "money laundering case",
    "your aadhaar is linked", "parcel contains drugs", "trai will block", "cyber crime department", "mumbai police", "sebi officer",
    "rbi officer", "arrest warrant", "stay on video call", "डिजिटल अरेस्ट", "सीबीआई", "गिरफ्तारी", "पुलिस अधिकारी", "अटक वॉरंट",
    "டிஜிட்டல் கைது", "காவல்",
  ],
  secrecy: [
    "do not tell anyone", "don't tell anyone", "keep this confidential", "don't inform family", "do not tell your family",
    "kisi ko mat batana", "किसी को मत बताना", "गुप्त रखें", "कोणालाही सांगू नका", "யாரிடமும் சொல்ல வேண்டாம்",
  ],
  remote_access: ["anydesk", "teamviewer", "quicksupport", "rustdesk", "screen share", "install this app", "apk file", ".apk"],
  kyc_threat: [
    "kyc expired", "kyc update", "account will be blocked", "account blocked", "pan not linked", "card blocked", "sim will be blocked",
    "electricity will be disconnected", "केवाईसी", "खाता बंद", "खाते बंद", "கணக்கு முடக்கப்படும்",
  ],
  insider_tip: [
    "insider tip", "operator stock", "sure shot", "sureshot", "multibagger", "jackpot stock", "upper circuit", "target hit",
    "big news coming", "pump", "operator game", "पक्की टिप", "शेयर टिप",
  ],
  defensive: [
    "beware of", "be aware", "is a scam", "is fraud", "never share", "report to 1930", "cybercrime.gov.in", "sebi warns",
    "awareness", "do not fall for", "सावधान", "सतर्क रहें", "धोखाधड़ी से बचें", "सावध", "எச்சரிக்கை",
  ],
};

/* ---------------- Aho–Corasick automaton (built once per Worker) ---------------- */

interface ACNode {
  next: Map<string, number>;
  fail: number;
  out: Array<{ family: FlagFamily; term: string }>;
}

export class AhoCorasick {
  private nodes: ACNode[] = [{ next: new Map(), fail: 0, out: [] }];

  constructor(dict: Record<FlagFamily, string[]>) {
    for (const fam of Object.keys(dict) as FlagFamily[]) for (const term of dict[fam]) this.insert(term, fam);
    this.build();
  }

  private insert(term: string, family: FlagFamily) {
    let s = 0;
    for (const ch of term) {
      let nx = this.nodes[s].next.get(ch);
      if (nx === undefined) {
        nx = this.nodes.length;
        this.nodes.push({ next: new Map(), fail: 0, out: [] });
        this.nodes[s].next.set(ch, nx);
      }
      s = nx;
    }
    this.nodes[s].out.push({ family, term });
  }

  private build() {
    const q: number[] = [];
    for (const [, c] of this.nodes[0].next) q.push(c);
    let head = 0;
    while (head < q.length) {
      const r = q[head++];
      for (const [ch, u] of this.nodes[r].next) {
        q.push(u);
        let f = this.nodes[r].fail;
        while (f && !this.nodes[f].next.has(ch)) f = this.nodes[f].fail;
        const cand = this.nodes[f].next.get(ch);
        this.nodes[u].fail = cand !== undefined && cand !== u ? cand : 0;
        this.nodes[u].out = this.nodes[u].out.concat(this.nodes[this.nodes[u].fail].out);
      }
    }
  }

  search(text: string): Array<{ family: FlagFamily; term: string; index: number }> {
    const hits: Array<{ family: FlagFamily; term: string; index: number }> = [];
    let s = 0;
    let i = 0;
    for (const ch of text) {
      while (s && !this.nodes[s].next.has(ch)) s = this.nodes[s].fail;
      s = this.nodes[s].next.get(ch) ?? 0;
      for (const o of this.nodes[s].out) hits.push({ ...o, index: i });
      i++;
    }
    return hits;
  }

  get size() {
    return this.nodes.length;
  }
}
