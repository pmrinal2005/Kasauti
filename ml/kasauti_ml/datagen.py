"""Synthetic multilingual training data for Kasauti's Laya fine-tune.

**Template-split, not random-split.** Every template id is assigned to exactly one of
train/val/test by a stable hash, and every surface variant of that template inherits the split.
A random split over generated rows would leak the same sentence with different numbers into both
halves and report an accuracy that means nothing; the held-out gold set therefore only ever
contains templates the model has never seen.

Three kinds of rows are produced:

* **positives** — the scam / promotion / intent patterns Kasauti must catch;
* **hard negatives** — legitimate education, advisories, corporate actions and benign questions
  that *use the same vocabulary* (the words "guaranteed", "risk", "VIP", "arrest" all appear in
  real warnings), which is what stops the model from becoming a keyword detector;
* **ambiguity rows** — deliberately soft targets, so RLCD has something to calibrate against
  instead of only near-one-hot labels.

Optional ``translate`` hook: pass ``--translate-cmd`` (an IndicTrans2 runner) and every English
template is expanded into the remaining target languages, keeping the template/split assignment.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LANGS: Sequence[str] = ("en", "hi", "mr", "ta")
ROMANIZED: Dict[str, str] = {"hi": "Hinglish (romanised Hindi)", "mr": "romanised Marathi"}

SLOTS: Dict[str, Sequence[str]] = {
    "group": ["Mega Profit Club", "Wealth Warriors VIP", "Bull Run Insiders", "Stock King Premium",
              "Nifty Jackpot Group", "SEBI Approved Traders", "Algo Profit Circle"],
    "pct": ["3%", "5%", "8%", "12%", "20%", "40%", "double", "10%", "7%"],
    "period": ["per day", "daily", "per week", "monthly", "in 30 days", "in 15 days", "per month"],
    "amount": ["₹5,000", "₹12,500", "₹2 lakh", "₹4.2 lakh", "₹50,000", "₹1,20,000", "₹9,999"],
    "upi": ["vipfund@ybl", "profits.pay@okaxis", "rbi.verify@paytm", "taxcell@oksbi",
            "sureshot.tips@okhdfcbank", "refund.dept@ybl"],
    "deadline": ["today only", "before 6 PM", "last 5 seats", "within 24 hours", "offer ends tonight",
                 "slot closes in 2 hours"],
    "officer": ["CBI officer", "Mumbai Cyber Cell officer", "Customs officer", "ED officer",
                "SEBI officer", "TRAI officer"],
    "app": ["AnyDesk", "TeamViewer", "QuickSupport", "RustDesk", "a remote support app"],
    "link": ["http://kyc-update.xyz", "https://sbi-verify.top/login", "bit.ly/claim-now",
             "http://ipo-allotment.club/form", "https://refund-process.site"],
    "company": ["Suryodaya Agrotech", "Nakshatra Infra", "Blue Ocean Exports", "Vidarbha Sugars",
                "Deccan Pharma", "Himalaya Cements"],
    "concept": ["NAV", "SIP", "expense ratio", "compounding", "diversification", "nomination",
                "volatility", "exit load", "index fund"],
    "city": ["Delhi", "Pune", "Nagpur", "Chennai", "Lucknow", "Kochi", "Jaipur", "Indore"],
    "bank": ["SBI", "HDFC Bank", "ICICI Bank", "Punjab National Bank", "Axis Bank", "Kotak Bank"],
    "reg": ["INH000009001", "INA000004521", "INH000012345", "INA000009876"],
}


@dataclass
class Template:
    id: str
    task: str
    text: Dict[str, str]
    triage: Optional[str] = None
    """Choice label for the task's triage question (must be a key of its criteria)."""
    followups: Dict[str, float] = field(default_factory=dict)
    """qid -> P(true) for that task's `noul` follow-ups."""
    soft: bool = False
    """Emit a softened target distribution (ambiguity rows) instead of near-one-hot."""
    negative: bool = False
    """Hard negative: legitimate text that shares the scam vocabulary."""


def _t(tid: str, task: str, triage: Optional[str], text: Dict[str, str],
       followups: Optional[Dict[str, float]] = None, soft: bool = False, negative: bool = False) -> Template:
    return Template(tid, task, text, triage, followups or {}, soft, negative)


# --------------------------------------------------------------------------------------
# claim_v1 — the scam checker
# --------------------------------------------------------------------------------------
CLAIM: List[Template] = [
    _t("claim.vip_guaranteed", "claim_v1", "investment_offer", {
        "en": "Congratulations! Join our VIP {group} on WhatsApp. Guaranteed {pct} {period}, zero risk. {deadline}. Pay {amount} to {upi}.",
        "hi": "बधाई! हमारे VIP {group} से जुड़ें। {pct} {period} की गारंटी, कोई रिस्क नहीं। {deadline}। {amount} भेजें {upi} पर।",
        "mr": "अभिनंदन! आमच्या VIP {group} मध्ये सामील व्हा. {pct} {period} हमी परतावा, धोका नाही. {deadline}. {amount} पाठवा {upi} वर.",
        "ta": "வாழ்த்துகள்! எங்கள் VIP {group} குழுவில் சேருங்கள். {pct} {period} உறுதியான லாபம், ஆபத்து இல்லை. {deadline}. {amount} அனுப்புங்கள் {upi}.",
    }, {"guaranteed": 0.97, "unregistered_group": 0.95, "pay_to_withdraw": 0.2}),

    _t("claim.pay_to_withdraw", "claim_v1", "payment_request", {
        "en": "Your profit of {amount} is ready. Pay 18% withdrawal tax to unlock your profit. Pay to {upi} and send the screenshot.",
        "hi": "आपका {amount} मुनाफ़ा तैयार है। निकालने के लिए 18% विथड्रॉल टैक्स भरें। {upi} पर भुगतान करें और स्क्रीनशॉट भेजें।",
        "mr": "तुमचा {amount} नफा तयार आहे. पैसे काढण्यासाठी 18% कर भरा. {upi} वर पाठवा आणि स्क्रीनशॉट पाठवा.",
        "ta": "உங்கள் {amount} லாபம் தயார். எடுக்க 18% வரி செலுத்துங்கள். {upi} க்கு அனுப்பி ஸ்கிரீன்ஷாட் அனுப்புங்கள்.",
    }, {"pay_to_withdraw": 0.97, "remote_app": 0.15}),

    _t("claim.digital_arrest", "claim_v1", "authority_threat", {
        "en": "This is {officer}. Your Aadhaar is linked to a money laundering case. Stay on the video call, do not tell anyone in your family, transfer {amount} for verification.",
        "hi": "मैं {officer} हूँ। आपका आधार मनी लॉन्ड्रिंग केस से जुड़ा है। वीडियो कॉल पर बने रहें, परिवार में किसी को न बताएं, वेरिफिकेशन के लिए {amount} ट्रांसफर करें।",
        "mr": "मी {officer} आहे. तुमचे आधार मनी लाँड्रिंग प्रकरणाशी जोडलेले आहे. व्हिडिओ कॉलवर राहा, कुटुंबातील कोणालाही सांगू नका, पडताळणीसाठी {amount} ट्रान्सफर करा.",
        "ta": "நான் {officer}. உங்கள் ஆதார் பணமோசடி வழக்குடன் இணைந்துள்ளது. வீடியோ காலில் இருங்கள், குடும்பத்தில் யாருக்கும் சொல்லாதீர்கள், சரிபார்ப்புக்கு {amount} அனுப்புங்கள்.",
    }, {"digital_arrest": 0.98, "secrecy": 0.97, "transfer": 0.95}),

    _t("claim.kyc_block", "claim_v1", "payment_request", {
        "en": "Dear customer, your KYC expired and your {bank} account will be blocked today. Update immediately: {link}",
        "hi": "प्रिय ग्राहक, आपका KYC expired है और {bank} खाता आज बंद हो जाएगा। तुरंत अपडेट करें: {link}",
        "mr": "प्रिय ग्राहक, तुमचे KYC संपले आहे आणि {bank} खाते आज बंद होईल. लगेच अपडेट करा: {link}",
        "ta": "அன்பான வாடிக்கையாளரே, உங்கள் KYC காலாவதியானது, {bank} கணக்கு இன்று முடக்கப்படும். உடனே புதுப்பிக்கவும்: {link}",
    }, {"remote_app": 0.1}),

    _t("claim.rumote_access_refund", "claim_v1", "payment_request", {
        "en": "To process your refund of {amount}, install {app} so our support team can access your screen. Keep the session open until the refund is credited.",
        "hi": "आपका {amount} रिफंड प्रोसेस करने के लिए {app} इंस्टॉल करें ताकि हमारी टीम आपकी स्क्रीन देख सके। रिफंड आने तक सेशन खुला रखें।",
        "mr": "तुमचा {amount} परतावा प्रक्रिया करण्यासाठी {app} इन्स्टॉल करा म्हणजे आमची टीम स्क्रीन पाहू शकेल. परतावा येईपर्यंत सेशन सुरू ठेवा.",
        "ta": "உங்கள் {amount} பணத்தைத் திரும்பப் பெற {app} ஐ நிறுவுங்கள், எங்கள் குழு திரையைப் பார்க்கும். பணம் வரும் வரை அமர்வைத் திறந்து வைக்கவும்.",
    }, {"remote_app": 0.96, "pay_to_withdraw": 0.35}),

    _t("claim.ipo_guaranteed", "claim_v1", "investment_offer", {
        "en": "IPO allotment guaranteed through our institutional account. {company} listing gain confirmed. Send {amount} to {upi} before {deadline}.",
        "hi": "हमारे इंस्टीट्यूशनल अकाउंट से IPO अलॉटमेंट गारंटीड। {company} की लिस्टिंग गेन पक्की। {deadline} से पहले {amount} भेजें {upi} पर।",
        "mr": "आमच्या इन्स्टिट्यूशनल खात्यातून IPO अलॉटमेंट हमी. {company} चा लिस्टिंग गेन नक्की. {deadline} आधी {amount} पाठवा {upi} वर.",
        "ta": "எங்கள் நிறுவனக் கணக்கு மூலம் IPO ஒதுக்கீடு உறுதி. {company} பட்டியல் லாபம் உறுதி. {deadline} க்கு முன் {amount} அனுப்புங்கள் {upi}.",
    }, {"guaranteed": 0.9, "unregistered_group": 0.8}),

    _t("claim.insider_tip", "claim_v1", "investment_offer", {
        "en": "Operator stock! {company} upper circuit tomorrow. Sure shot multibagger tip from our insider. Join the paid group, {deadline}.",
        "hi": "ऑपरेटर स्टॉक! {company} कल अपर सर्किट। हमारे इनसाइडर से पक्की मल्टीबैगर टिप। पेड ग्रुप जॉइन करें, {deadline}।",
        "mr": "ऑपरेटर स्टॉक! {company} उद्या अप्पर सर्किट. आमच्या इनसायडरकडून पक्की मल्टीबॅगर टीप. पेड ग्रुप जॉईन करा, {deadline}.",
        "ta": "ஆபரேட்டர் பங்கு! {company} நாளை அப்பர் சர்க்கிட். எங்கள் உள்நபரின் உறுதியான டிப். கட்டணக் குழுவில் சேருங்கள், {deadline}.",
    }, {"guaranteed": 0.55, "unregistered_group": 0.85}),

    _t("claim.secrecy_pressure", "claim_v1", "authority_threat", {
        "en": "{officer} here. Do not tell anyone about this call, not even your wife. The matter is confidential and you will be arrested if you disclose it.",
        "hi": "{officer} बोल रहा हूँ। इस कॉल के बारे में किसी को न बताएं, पत्नी को भी नहीं। मामला गोपनीय है, बताने पर गिरफ्तारी होगी।",
        "mr": "{officer} बोलत आहे. या कॉलबद्दल कोणालाही सांगू नका, बायकोलाही नाही. प्रकरण गोपनीय आहे, सांगितल्यास अटक होईल.",
        "ta": "{officer} பேசுகிறேன். இந்த அழைப்பைப் பற்றி யாரிடமும் சொல்லாதீர்கள், மனைவியிடமும் சொல்லாதீர்கள். விஷயம் ரகசியம், சொன்னால் கைது.",
    }, {"secrecy": 0.97, "digital_arrest": 0.8, "transfer": 0.4}),

    # ---- hard negatives: legitimate text that shares the vocabulary ----
    _t("claim.neg.sebi_warning", "claim_v1", "education", {
        "en": "SEBI warns investors: beware of unregistered advisers promising guaranteed returns. Report such entities on SCORES or call 1930.",
        "hi": "SEBI की चेतावनी: गारंटीड रिटर्न का वादा करने वाले अपंजीकृत सलाहकारों से सावधान रहें। SCORES पर शिकायत करें या 1930 पर कॉल करें।",
        "mr": "SEBI चा इशारा: हमी परतावा देणाऱ्या नोंदणी नसलेल्या सल्लागारांपासून सावध राहा. SCORES वर तक्रार करा किंवा 1930 वर कॉल करा.",
        "ta": "SEBI எச்சரிக்கை: உறுதியான லாபம் தருவதாகக் கூறும் பதிவு செய்யாத ஆலோசகர்களைக் கவனியுங்கள். SCORES இல் புகார் அளியுங்கள் அல்லது 1930 க்கு அழையுங்கள்.",
    }, {"selling": 0.02}, negative=True),

    _t("claim.neg.education", "claim_v1", "education", {
        "en": "What is {concept}? In this explainer we show how {concept} works, with a worked example and no product recommendation.",
        "hi": "{concept} क्या है? इस वीडियो में हम {concept} का मतलब उदाहरण के साथ समझाते हैं, किसी प्रोडक्ट की सिफारिश नहीं करते।",
        "mr": "{concept} म्हणजे काय? या व्हिडिओत आम्ही {concept} चा अर्थ उदाहरणासह समजावून सांगतो, कोणत्याही उत्पादनाची शिफारस करत नाही.",
        "ta": "{concept} என்றால் என்ன? இந்த வீடியோவில் {concept} எப்படி வேலை செய்கிறது என்பதை உதாரணத்துடன் விளக்குகிறோம், எந்த பொருளையும் பரிந்துரைக்கவில்லை.",
    }, {"selling": 0.03}, negative=True),

    _t("claim.neg.corporate_action", "claim_v1", "other", {
        "en": "{company} board has recommended a final dividend of ₹4 per share. The record date is 12 August and the AGM notice is enclosed for adoption of accounts.",
        "hi": "{company} बोर्ड ने ₹4 प्रति शेयर फाइनल डिविडेंड की सिफारिश की है। रिकॉर्ड डेट 12 अगस्त है और खातों के अनुमोदन का AGM नोटिस संलग्न है।",
        "mr": "{company} बोर्डाने ₹4 प्रति शेअर अंतिम लाभांश शिफारस केला आहे. रेकॉर्ड डेट 12 ऑगस्ट आहे आणि खात्यांच्या मंजुरीसाठी AGM सूचना जोडली आहे.",
        "ta": "{company} வாரியம் ஒரு பங்குக்கு ₹4 இறுதி ஈவுத்தொகையை பரிந்துரைத்துள்ளது. பதிவு தேதி ஆகஸ்ட் 12, கணக்குகளை ஏற்க AGM அறிவிப்பு இணைக்கப்பட்டுள்ளது.",
    }, {"selling": 0.02}, negative=True),

    _t("claim.neg.regulator_advisory_bank", "claim_v1", "education", {
        "en": "Advisory from {bank}: we never ask for OTP, PIN or CVV on calls, and we never send KYC links by SMS. Forward suspicious messages to 1909.",
        "hi": "{bank} की सलाह: हम कभी कॉल पर OTP, PIN या CVV नहीं मांगते और SMS से KYC लिंक नहीं भेजते। संदिग्ध संदेश 1909 पर भेजें।",
        "mr": "{bank} चा सल्ला: आम्ही कधीही कॉलवर OTP, PIN किंवा CVV मागत नाही आणि SMS ने KYC लिंक पाठवत नाही. संशयास्पद संदेश 1909 वर पाठवा.",
        "ta": "{bank} அறிவுறுத்தல்: அழைப்பில் OTP, PIN அல்லது CVV கேட்க மாட்டோம், SMS மூலம் KYC இணைப்பு அனுப்ப மாட்டோம். சந்தேகமான செய்திகளை 1909 க்கு அனுப்புங்கள்.",
    }, {"selling": 0.02}, negative=True),

    _t("claim.neg.family_forward", "claim_v1", "other", {
        "en": "Sharing the {concept} article you asked for. Nothing to buy, just the explanation with the formula and two examples.",
        "hi": "आपने जो {concept} आर्टिकल मांगा था, वह भेज रहा हूँ। कुछ खरीदना नहीं है, सिर्फ फॉर्मूला और दो उदाहरण के साथ समझाया है।",
        "mr": "तुम्ही मागितलेला {concept} लेख पाठवत आहे. काही खरेदी करायची नाही, फक्त सूत्र आणि दोन उदाहरणांसह स्पष्टीकरण.",
        "ta": "நீங்கள் கேட்ட {concept} கட்டுரையை அனுப்புகிறேன். வாங்க எதுவும் இல்லை, சூத்திரம் மற்றும் இரண்டு உதாரணங்களுடன் விளக்கம்.",
    }, {"selling": 0.04}, negative=True),

    # ---- deliberately ambiguous rows: soft targets for calibration ----
    _t("claim.amb.mentions_words", "claim_v1", "investment_offer", {
        "en": "Our channel discusses guaranteed return products and VIP group mechanics so you can recognise them in the wild.",
        "hi": "हमारा चैनल गारंटीड रिटर्न प्रोडक्ट और VIP ग्रुप के तरीके बताता है ताकि आप उन्हें पहचान सकें।",
        "mr": "आमचा चॅनल हमी परतावा उत्पादने आणि VIP ग्रुपच्या पद्धती समजावून सांगतो जेणेकरून तुम्ही ओळखू शकाल.",
        "ta": "உறுதியான லாப தயாரிப்புகள், VIP குழு முறைகளை எங்கள் சேனல் விளக்குகிறது, அவற்றை அடையாளம் காண.",
    }, {"guaranteed": 0.5, "unregistered_group": 0.4, "pay_to_withdraw": 0.2}, soft=True),

    _t("claim.amb.partner_offer", "claim_v1", "investment_offer", {
        "en": "Partner with us for portfolio management. Returns are market-linked and past performance does not indicate future results. Registration {reg}.",
        "hi": "पोर्टफोलियो मैनेजमेंट के लिए हमसे जुड़ें। रिटर्न बाज़ार से जुड़ा है, पिछला प्रदर्शन भविष्य की गारंटी नहीं। रजिस्ट्रेशन {reg}।",
        "mr": "पोर्टफोलिओ व्यवस्थापनासाठी आमच्याशी जोडा. परतावा बाजाराशी जोडलेला आहे, पूर्वीची कामगिरी भविष्याची हमी नाही. नोंदणी {reg}.",
        "ta": "போர்ட்ஃபோலியோ மேலாண்மைக்கு எங்களுடன் இணையுங்கள். லாபம் சந்தையுடன் இணைந்தது, கடந்த செயல்திறன் எதிர்கால உத்தரவாதம் அல்ல. பதிவு {reg}.",
    }, {"guaranteed": 0.15, "unregistered_group": 0.3}, soft=True),
]

# --------------------------------------------------------------------------------------
# lens_v1 — promotion vs education
# --------------------------------------------------------------------------------------
LENS: List[Template] = [
    _t("lens.pushes_security", "lens_v1", "pushes_security", {
        "en": "Buy {company} now. Target ₹480 in 30 days, stop loss ₹200. Join my channel for the next sure-shot call, {deadline}.",
        "hi": "{company} अभी खरीदें। 30 दिन में टारगेट ₹480, स्टॉप लॉस ₹200। अगली पक्की कॉल के लिए चैनल जॉइन करें, {deadline}।",
        "mr": "{company} आता खरेदा. 30 दिवसांत टार्गेट ₹480, स्टॉप लॉस ₹200. पुढील पक्क्या कॉलसाठी चॅनल जॉईन करा, {deadline}.",
        "ta": "{company} இப்போது வாங்குங்கள். 30 நாட்களில் இலக்கு ₹480, ஸ்டாப் லாஸ் ₹200. அடுத்த உறுதியான காலுக்கு சேனலில் சேருங்கள், {deadline}.",
    }, {"target_claim": 0.97, "disclosure": 0.05}),

    _t("lens.teaches", "lens_v1", "teaches_concept", {
        "en": "Understanding {concept}: what it is, how it is calculated and why it matters over long periods. No recommendations, no products.",
        "hi": "{concept} को समझें: यह क्या है, कैसे गणना होती है और लंबे समय में क्यों मायने रखता है। कोई सिफारिश नहीं, कोई प्रोडक्ट नहीं।",
        "mr": "{concept} समजून घ्या: ते काय आहे, कसे मोजले जाते आणि दीर्घकाळात का महत्त्वाचे आहे. कोणतीही शिफारस नाही, उत्पादन नाही.",
        "ta": "{concept} பற்றி புரிந்து கொள்ளுங்கள்: அது என்ன, எப்படி கணக்கிடப்படுகிறது, நீண்ட காலத்திற்கு ஏன் முக்கியம். பரிந்துரைகள் இல்லை.",
    }, {"disclosure": 0.1, "selling": 0.03}),

    _t("lens.fear", "lens_v1", "fear_selling", {
        "en": "The market is about to crash. Only people in my protected group will survive. Move your money tonight, {deadline}.",
        "hi": "बाज़ार क्रैश होने वाला है। सिर्फ मेरे प्रोटेक्टेड ग्रुप वाले बचेंगे। आज रात पैसा शिफ्ट करें, {deadline}।",
        "mr": "बाजार कोसळणार आहे. फक्त माझ्या संरक्षित ग्रुपमधले वाचतील. आज रात्री पैसे हलवा, {deadline}.",
        "ta": "சந்தை வெடிக்கப்போகிறது. எனது பாதுகாப்பு குழுவில் உள்ளவர்கள் மட்டுமே தப்பிப்பார்கள். இன்று இரவே பணத்தை மாற்றுங்கள், {deadline}.",
    }, {"guaranteed": 0.7, "urgency": 0.95}),

    _t("lens.testimonial", "lens_v1", "testimonial_lure", {
        "en": "Look at my students' profit screenshots: {pct} {period}. Same system, private group, limited members. Registration {reg} available on request.",
        "hi": "मेरे स्टूडेंट्स के प्रॉफिट स्क्रीनशॉट देखें: {pct} {period}। वही सिस्टम, प्राइवेट ग्रुप, सीमित मेंबर। रजिस्ट्रेशन {reg} मांगने पर।",
        "mr": "माझ्या विद्यार्थ्यांचे नफा स्क्रीनशॉट पहा: {pct} {period}. तेच सिस्टम, खाजगी ग्रुप, मर्यादित सदस्य. नोंदणी {reg} मागणी केल्यास.",
        "ta": "எனது மாணவர்களின் லாப ஸ்கிரீன்ஷாட்களைப் பாருங்கள்: {pct} {period}. அதே அமைப்பு, தனிக் குழு, குறைந்த உறுப்பினர்கள். பதிவு {reg}.",
    }, {"target_claim": 0.85, "unregistered_group": 0.8}),

    _t("lens.push_product", "lens_v1", "pushes_product", {
        "en": "Open your account with my partner broker links and get a fee waiver. Ethical investing needs the right platform, {deadline}.",
        "hi": "मेरे पार्टनर ब्रोकर लिंक से अकाउंट खोलें और फीस छूट पाएं। सही निवेश के लिए सही प्लेटफॉर्म ज़रूरी है, {deadline}।",
        "mr": "माझ्या पार्टनर ब्रोकर लिंकने खाते उघडा आणि फी सवलत मिळवा. योग्य गुंतवणुकीसाठी योग्य प्लॅटफॉर्म आवश्यक, {deadline}.",
        "ta": "எனது கூட்டாளர் தரகரிடம் கணக்கு திறந்து கட்டணம் தள்ளுபடி பெறுங்கள். சரியான முதலீட்டுக்கு சரியான தளம் தேவை, {deadline}.",
    }, {"guaranteed": 0.2, "disclosure": 0.1}),

    _t("lens.neg.research", "lens_v1", "reports_facts", {
        "en": "Q2 results: {company} revenue up 8% year on year, margins flat. This is a factual summary of the filing, not investment advice.",
        "hi": "दूसरी तिमाही नतीजे: {company} का रेवेन्यू सालाना 8% बढ़ा, मार्जिन सपाट। यह फाइलिंग का तथ्यात्मक सारांश है, निवेश सलाह नहीं।",
        "mr": "दुसऱ्या तिमाहीचे निकाल: {company} चा महसूल वर्षभरात 8% वाढला, मार्जिन स्थिर. हे फाइलिंगचे तथ्यात्मक सारांश आहे, गुंतवणूक सल्ला नाही.",
        "ta": "இரண்டாம் காலாண்டு முடிவுகள்: {company} வருவாய் ஆண்டுக்கு 8% உயர்வு. இது ஆவணத்தின் உண்மைச் சுருக்கம், முதலீட்டு ஆலோசனை அல்ல.",
    }, {"selling": 0.03, "disclosure": 0.2}, negative=True),

    _t("lens.neg.awareness_reel", "lens_v1", "teaches_concept", {
        "en": "Three red flags of fradulent advisers: guaranteed returns, no registration number, and pressure to pay into a personal UPI ID. Verify on SEBI's website.",
        "hi": "धोखाधड़ी करने वाले सलाहकारों के तीन संकेत: गारंटीड रिटर्न, रजिस्ट्रेशन नंबर नहीं, और निजी UPI पर पैसा डालने का दबाव। SEBI की वेबसाइट पर जांचें।",
        "mr": "फसव्या सल्लागारांची तीन लक्षणे: हमी परतावा, नोंदणी क्रमांक नाही, आणि वैयक्तिक UPI वर पैसे भरण्याचा दबाव. SEBI च्या वेबसाइटवर तपासा.",
        "ta": "மோசடி ஆலோசகர்களின் மூன்று அறிகுறிகள்: உறுதியான லாபம், பதிவு எண் இல்லை, தனிப்பட்ட UPI க்கு பணம் செலுத்தும் அழுத்தம். SEBI தளத்தில் சரிபார்க்கவும்.",
    }, {"selling": 0.03, "disclosure": 0.3}, negative=True),

    _t("lens.amb.paid_promo", "lens_v1", "pushes_product", {
        "en": "This video is a paid promotion for {company}'s app. The app is a platform; investing carries market risk and no return is promised.",
        "hi": "यह वीडियो {company} के ऐप का पेड प्रोमोशन है। ऐप एक प्लेटफॉर्म है; निवेश में बाजार जोखिम है और किसी रिटर्न का वादा नहीं है।",
        "mr": "हा व्हिडिओ {company} च्या ॲपचा पेड प्रोमोशन आहे. ॲप एक प्लॅटफॉर्म आहे; गुंतवणुकीत बाजार धोका असतो आणि परताव्याचे वचन नाही.",
        "ta": "இது {company} செயலியின் கட்டண விளம்பரம். செயலி ஒரு தளம்; முதலீட்டில் சந்தை ஆபத்து உள்ளது, லாபம் உறுதி இல்லை.",
    }, {"guaranteed": 0.1, "disclosure": 0.9}, soft=True),
]

# --------------------------------------------------------------------------------------
# intent_concepts_v1 / report_v1 / impersonation_v1 / resolution_v1
# --------------------------------------------------------------------------------------
INTENT: List[Template] = [
    _t("intent.sip", "intent_concepts_v1", "basics", {
        "en": "{concept} kya hota hai? Please explain simply, I am new to mutual funds.",
        "hi": "{concept} क्या होता है? आसान भाषा में बताइए, मैं नया हूँ।",
        "mr": "{concept} म्हणजे काय? सोप्या भाषेत सांगा, मी नवीन आहे.",
        "ta": "{concept} என்பது என்ன? எளிமையாக விளக்குங்கள், நான் புதியவன்.",
    }, {"nav": 0.12, "sip": 0.9, "compounding": 0.15}),

    _t("intent.costs", "intent_concepts_v1", "costs", {
        "en": "How much does a fund charge every year? What is the {concept} and how does it affect my returns?",
        "hi": "फंड हर साल कितना चार्ज लेता है? {concept} क्या है और मेरे रिटर्न पर कैसे असर डालता है?",
        "mr": "फंड दरवर्षी किती शुल्क घेतो? {concept} काय आहे आणि माझ्या परताव्यावर कसा परिणाम करतो?",
        "ta": "நிதி ஆண்டுதோறும் எவ்வளவு கட்டணம் எடுக்கிறது? {concept} என்ன, அது லாபத்தை எப்படி பாதிக்கிறது?",
    }, {"expense_ratio": 0.9, "exit_load": 0.35}),

    _t("intent.redflag", "intent_concepts_v1", "red_flags", {
        "en": "Someone added me to a group promising {pct} {period}. Is this a {concept} case? Should I be worried?",
        "hi": "किसी ने मुझे ग्रुप में जोड़ा जो {pct} {period} का वादा कर रहा है। क्या यह {concept} है? मुझे डरना चाहिए?",
        "mr": "कोणीतरी मला ग्रुपमध्ये टाकले जे {pct} {period} चे वचन देत आहे. हे {concept} आहे का? मी घाबरावे का?",
        "ta": "ஒருவர் என்னை {pct} {period} வாக்குறுதி தரும் குழுவில் சேர்த்தார். இது {concept} ஆகுமா? நான் பயப்பட வேண்டுமா?",
    }, {"guaranteed": 0.85, "vip_group": 0.9}),

    _t("intent.safety", "intent_concepts_v1", "safety", {
        "en": "I transferred money to a fake app. How do I report it and what is {concept}?",
        "hi": "मैंने नकली ऐप पर पैसे भेज दिए। शिकायत कैसे करूँ और {concept} क्या है?",
        "mr": "मी बनावट ॲपवर पैसे पाठवले. तक्रार कशी करू आणि {concept} काय आहे?",
        "ta": "போலி செயலிக்கு பணம் அனுப்பிவிட்டேன். எப்படி புகார் அளிப்பது, {concept} என்ன?",
    }, {"nomination": 0.1, "reporting": 0.9}),

    _t("intent.amb.mixed", "intent_concepts_v1", "unclear", {
        "en": "Please tell me about {concept} and also the {concept} thing, I forgot the exact word.",
        "hi": "{concept} के बारे में बताइए और वह {concept} वाली बात भी, शब्द याद नहीं आ रहा।",
        "mr": "{concept} बद्दल सांगा आणि ती {concept} गोष्ट पण, शब्द आठवत नाही.",
        "ta": "{concept} பற்றி சொல்லுங்கள், அந்த {concept} விஷயமும், வார்த்தை நினைவில் இல்லை.",
    }, {"nav": 0.4, "sip": 0.4, "compounding": 0.4}, soft=True),
]

REPORT: List[Template] = [
    _t("report.trading_app", "report_v1", "trading_app", {
        "en": "A {city} based app showed {pct} {period} profits but now asks for a 20% tax before withdrawal. My friend lost {amount}.",
        "hi": "{city} की एक ऐप {pct} {period} प्रॉफिट दिखाती थी, अब निकालने से पहले 20% टैक्स मांगती है। मेरे दोस्त के {amount} डूब गए।",
        "mr": "{city} मधील एक ॲप {pct} {period} नफा दाखवत होते, आता काढण्यापूर्वी 20% कर मागत आहे. माझ्या मित्राचे {amount} गेले.",
        "ta": "{city} செயலி {pct} {period} லாபம் காட்டியது, இப்போது எடுக்க 20% வரி கேட்கிறது. என் நண்பரின் {amount} இழந்தது.",
    }, {"pay_to_withdraw": 0.95, "remote_app": 0.2}),

    _t("report.impersonation", "report_v1", "impersonation", {
        "en": "Caller said he was from {officer} and kept my uncle on video for two hours, told him not to inform family, then demanded {amount}.",
        "hi": "कॉलर ने कहा वह {officer} से है, चाचा को दो घंटे वीडियो पर रखा, परिवार को न बताने को कहा, फिर {amount} मांगे।",
        "mr": "कॉलर म्हणाला तो {officer} आहे, काकांना दोन तास व्हिडिओवर ठेवले, कुटुंबाला न सांगा म्हणाले, नंतर {amount} मागितले.",
        "ta": "அழைப்பவர் {officer} என்றார், மாமாவை இரண்டு மணி நேரம் வீடியோவில் வைத்தார், குடும்பத்திற்கு சொல்ல வேண்டாம் என்றார், பின் {amount} கேட்டார்.",
    }, {"digital_arrest": 0.95, "secrecy": 0.92}),

    _t("report.vip_group", "report_v1", "vip_group", {
        "en": "Our self-help group members paid {amount} into {upi} after a VIP group promised guaranteed {pct} {period}.",
        "hi": "हमारे स्वयं सहायता समूह की महिलाओं ने {upi} पर {amount} जमा किया, VIP ग्रुप ने {pct} {period} गारंटी दी थी।",
        "mr": "आमच्या बचत गटातील महिलांनी {upi} वर {amount} भरले, VIP ग्रुपने {pct} {period} हमी दिली होती.",
        "ta": "எங்கள் சுய உதவிக் குழு உறுப்பினர்கள் {upi} க்கு {amount} செலுத்தினர், VIP குழு {pct} {period} உறுதி அளித்தது.",
    }, {"guaranteed": 0.93, "upi": 0.9}),

    _t("report.kyc", "report_v1", "kyc_phishing", {
        "en": "Received a message that my {bank} KYC will expire and account will be blocked, with a link {link}. I clicked it and then got an OTP call.",
        "hi": "मुझे संदेश आया कि {bank} का KYC खत्म हो जाएगा और खाता बंद होगा, लिंक {link} के साथ। क्लिक करने के बाद OTP कॉल आया।",
        "mr": "मला संदेश आला की {bank} चे KYC संपेल आणि खाते बंद होईल, लिंक {link} सह. क्लिक केल्यावर OTP कॉल आला.",
        "ta": "{bank} KYC முடிவடையும், கணக்கு முடக்கப்படும் என்று இணைப்பு {link} உடன் செய்தி வந்தது. கிளிக் செய்தபின் OTP அழைப்பு வந்தது.",
    }, {"link": 0.95, "otp": 0.85}),

    _t("report.job_fee", "report_v1", "job_loan", {
        "en": "A company asked for {amount} as a refundable security deposit for a work-from-home job and then stopped replying in {city}.",
        "hi": "एक कंपनी ने वर्क-फ्रॉम-होम नौकरी के लिए {amount} रिफंडेबल डिपॉज़िट मांगा और फिर {city} में जवाब देना बंद कर दिया।",
        "mr": "एका कंपनीने वर्क-फ्रॉम-होम नोकरीसाठी {amount} परतावा जमा मागितली आणि नंतर {city} मध्ये उत्तर देणे बंद केले.",
        "ta": "வீட்டிலிருந்து வேலைக்கு {amount} திரும்பப் பெறக்கூடிய வைப்பு என்று ஒரு நிறுவனம் கேட்டது, பின்னர் {city} இல் பதில் இல்லை.",
    }, {"advance_fee": 0.95, "personal_docs": 0.3}),

    _t("report.neg.misunderstanding", "report_v1", "other", {
        "en": "Updated my bank details after a branch visit and the message looked suspicious, but the branch manager confirmed it was genuine.",
        "hi": "शाखा जाकर बैंक विवरण अपडेट किया और संदेश संदिग्ध लगा, पर मैनेजर ने पुष्टि की यह असली था।",
        "mr": "शाखेत जाऊन बँक तपशील अपडेट केले आणि संदेश संशयास्पद वाटला, पण व्यवस्थापकाने खरे असल्याचे सांगितले.",
        "ta": "கிளைக்குச் சென்று வங்கி விவரங்களைப் புதுப்பித்தேன், செய்தி சந்தேகமாக இருந்தது, ஆனால் மேலாளர் உண்மை என்று உறுதிப்படுத்தினார்.",
    }, {"advance_fee": 0.05}, negative=True),
]

IMPERSONATION: List[Template] = [
    _t("imp.police", "impersonation_v1", "police", {
        "en": "The caller said he was a {officer} and told me a parcel in my name had narcotics, and I must stay on video for verification.",
        "hi": "कॉलर ने कहा वह {officer} है और मेरे नाम के पार्सल में ड्रग्स मिला है, वेरिफिकेशन के लिए वीडियो पर रुकना होगा।",
        "mr": "कॉलर म्हणाला तो {officer} आहे आणि माझ्या नावाच्या पार्सलमध्ये ड्रग्स सापडले, पडताळणीसाठी व्हिडिओवर राहावे लागेल.",
        "ta": "அழைத்தவர் {officer} என்றார், என் பெயரில் உள்ள பார்சலில் போதைப்பொருள் உள்ளது, சரிபார்ப்புக்கு வீடியோவில் இருக்க வேண்டும் என்றார்.",
    }, {"digital_arrest": 0.97, "transfer": 0.6}),

    _t("imp.bank", "impersonation_v1", "bank", {
        "en": "{bank} support called and asked me to share the OTP to reverse a wrong debit, and warned the account would be blocked within minutes.",
        "hi": "{bank} सपोर्ट ने कॉल कर के गलत डेबिट वापस करने के लिए OTP मांगा और कहा मिनटों में खाता बंद हो जाएगा।",
        "mr": "{bank} सपोर्टने कॉल करून चुकीचा डेबिट परत करण्यासाठी OTP मागितला आणि मिनिटांत खाते बंद होईल म्हणाले.",
        "ta": "{bank} ஆதரவு என்று அழைத்து தவறான பற்றை திரும்பச் செய்ய OTP கேட்டது, நிமிடங்களில் கணக்கு முடக்கப்படும் என்று எச்சரித்தது.",
    }, {"otp": 0.97, "urgency": 0.9}),

    _t("imp.regulator", "impersonation_v1", "regulator", {
        "en": "Someone claiming to be from SEBI said my demat account is under investigation, asked me to install {app} and keep it confidential from family.",
        "hi": "SEBI से होने का दावा करने वाले ने कहा मेरा डीमैट खाता जांच में है, {app} इंस्टॉल करने को कहा और परिवार से गोपनीय रखने को कहा।",
        "mr": "SEBI चा असल्याचा दावा करणारा म्हणाला माझे डीमॅट खाते तपासात आहे, {app} इन्स्टॉल करा आणि कुटुंबापासून गुप्त ठेवा.",
        "ta": "SEBI எனக் கூறியவர் என் டீமேட் கணக்கு விசாரணையில் உள்ளது என்றார், {app} நிறுவச் சொன்னார், குடும்பத்திடம் ரகசியம் வைக்கச் சொன்னார்.",
    }, {"secrecy": 0.93, "remote_app": 0.9}),

    _t("imp.neg.bank_genuine", "impersonation_v1", "bank", {
        "en": "My bank's fraud team called from the number printed on my card and asked me to confirm the last transaction I had made; they did not ask for any OTP.",
        "hi": "बैंक की फ्रॉड टीम ने कार्ड पर छपे नंबर से कॉल कर के पिछला लेनदेन पुष्टि करने को कहा; OTP नहीं मांगा।",
        "mr": "बँकेच्या फसवणूक विभागाने कार्डवरील क्रमांकावरून कॉल करून शेवटचा व्यवहार पुष्टी करायला सांगितला; OTP मागितला नाही.",
        "ta": "வங்கியின் மோசடி குழு அட்டையில் உள்ள எண்ணிலிருந்து அழைத்து கடைசி பரிவர்த்தனையை உறுதிப்படுத்தக் கேட்டது; OTP கேட்கவில்லை.",
    }, {"otp": 0.05, "urgency": 0.2}, negative=True),
]

RESOLUTION: List[Template] = [
    _t("res.dividend", "resolution_v1", "dividend", {
        "en": "{company} has fixed the record date for the final dividend. Shareholders on the register on that date will receive ₹4 per share, subject to deduction of tax at source.",
        "hi": "{company} ने फाइनल डिविडेंड की रिकॉर्ड डेट तय की है। उस तारीख को रजिस्टर में मौजूद शेयरधारकों को ₹4 प्रति शेयर मिलेगा, TDS के अधीन।",
        "mr": "{company} ने अंतिम लाभांशाची रेकॉर्ड डेट ठरवली आहे. त्या तारखेला रजिस्टरवर असलेल्या भागधारकांना ₹4 प्रति शेअर मिळेल, TDS लागू.",
        "ta": "{company} இறுதி ஈவுத்தொகைக்கான பதிவு தேதியை நிர்ணயித்துள்ளது. அந்த தேதியில் பங்குதாரர்கள் ₹4 பெறுவார்கள், வரி பிடித்தம் உண்டு.",
    }, {"record_date_effect": 0.9, "tax_note": 0.85}),

    _t("res.bonus", "resolution_v1", "bonus_split", {
        "en": "{company} proposes a 1:2 bonus issue. The number of shares held will increase and the face value will be split accordingly; no money is payable by shareholders.",
        "hi": "{company} 1:2 बोनस इश्यू का प्रस्ताव करता है। शेयरों की संख्या बढ़ेगी और फेस वैल्यू विभाजित होगी; शेयरधारकों को पैसे देने नहीं हैं।",
        "mr": "{company} 1:2 बोनस इश्यूचा प्रस्ताव ठेवतो. शेअर्सची संख्या वाढेल आणि दर्शनी किंमत विभागली जाईल; भागधारकांना पैसे द्यावे लागणार नाहीत.",
        "ta": "{company} 1:2 போனஸ் வெளியீட்டை முன்மொழிகிறது. பங்குகளின் எண்ணிக்கை அதிகரிக்கும், முக மதிப்பு பிரிக்கப்படும்; பணம் செலுத்த வேண்டியதில்லை.",
    }, {"share_count_change": 0.93, "dilution": 0.1}),

    _t("res.merger", "resolution_v1", "merger", {
        "en": "The scheme of arrangement provides for swap of {company} shares in the ratio 4:1 of the transferee company, or a cash exit price for eligible shareholders.",
        "hi": "योजना के तहत {company} के शेयर 4:1 अनुपात में नई कंपनी के शेयरों में बदलेंगे, या पात्र शेयरधारकों को नकद एग्ज़िट मिलेगा।",
        "mr": "योजनेनुसार {company} चे शेअर्स 4:1 प्रमाणात नव्या कंपनीच्या शेअर्समध्ये बदलतील, किंवा पात्र भागधारकांना रोख एग्झिट मिळेल.",
        "ta": "திட்டத்தின் படி {company} பங்குகள் 4:1 விகிதத்தில் மாற்றப்படும், அல்லது தகுதியுள்ள பங்குதாரர்களுக்கு பண வெளியேறு வழி உள்ளது.",
    }, {"share_exchange": 0.95, "exit_option": 0.8}),

    _t("res.remuneration", "resolution_v1", "remuneration", {
        "en": "Approval is sought for payment of remuneration to the managing director who is a promoter, up to a ceiling of ₹2 crore per year, as a related-party transaction.",
        "hi": "प्रोमोटर मैनेजिंग डायरेक्टर को ₹2 करोड़ प्रति वर्ष तक पारिश्रमिक देने की स्वीकृति मांगी गई है, संबंधित पक्ष लेनदेन के रूप में।",
        "mr": "प्रोमोटर असलेल्या व्यवस्थापकीय संचालकाला वर्षाला ₹2 कोटीपर्यंत मानधन देण्यास मान्यता मागितली आहे, संबंधित पक्ष व्यवहार म्हणून.",
        "ta": "ப்ரோமோட்டரான நிர்வாக இயக்குநருக்கு ஆண்டுக்கு ₹2 கோடி வரை ஊதியம் வழங்க ஒப்புதல் கோரப்படுகிறது, தொடர்புடைய தரப்பு பரிவர்த்தனையாக.",
    }, {"related_party": 0.93, "cap_stated": 0.95}),

    _t("res.neg.routine", "resolution_v1", "routine", {
        "en": "The board recommends adoption of the audited financial statements for the year ended 31 March and the appointment of the statutory auditors, with no qualification.",
        "hi": "बोर्ड 31 मार्च को समाप्त वर्ष के ऑडिटेड वित्तीय विवरण और स्टेट्यूटरी ऑडिटर की नियुक्ति के अनुमोदन की सिफारिश करता है, बिना किसी योग्यता के।",
        "mr": "बोर्ड 31 मार्चला संपलेल्या वर्षाच्या ऑडिट केलेल्या आर्थिक विवरणपत्रांचे मंजूर आणि स्टॅच्युटरी ऑडिटरची नेमणूक शिफारस करतो, कोणत्याही अटीशिवाय.",
        "ta": "மார்ச் 31 இல் முடிந்த ஆண்டுக்கான தணிக்கை செய்யப்பட்ட நிதிநிலை அறிக்கையை ஏற்கவும், சட்டப்பூர்வ தணிக்கையாளரை நியமிக்கவும் வாரியம் பரிந்துரைக்கிறது.",
    }, {"auditor_change": 0.07, "accounts_adopted": 0.9}, negative=True),
]

TEMPLATES_BY_TASK: Dict[str, List[Template]] = {
    "claim_v1": CLAIM, "lens_v1": LENS, "intent_concepts_v1": INTENT,
    "report_v1": REPORT, "impersonation_v1": IMPERSONATION, "resolution_v1": RESOLUTION,
}

# --------------------------------------------------------------------------------------
# surface variants
# --------------------------------------------------------------------------------------
_PREFIX = ["Fwd as received:", "Forwarded many times", "[{city} group]", "From: +91 98xxxxxxx1"]
_SUFFIX = [" Please advise.", " Is this safe?", " Someone sent this in our colony group.", ""]
_OBFUSC = [("\u200b", "zero-width space"), ("\u200d", "zero-width joiner")]


def _obfuscate(text: str, rng: random.Random) -> str:
    """Insert a zero-width character and swap one Latin letter for a Cyrillic homoglyph."""
    if not text:
        return text
    pos = rng.randrange(len(text))
    text = text[:pos] + "\u200b" + text[pos:]
    homoglyphs = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у"}
    for src, dst in homoglyphs.items():
        idx = text.find(src)
        if idx >= 0:
            text = text[:idx] + dst + text[idx + 1:]
            break
    return text


def _romanize_hints(text: str, rng: random.Random, lang: str) -> str:
    """Cheap code-mixing: add romanised fillers typical of forwarded WhatsApp text."""
    fillers = {
        "hi": ["bhai", "jaldi", "turant", "paisa", "confirm karo"],
        "mr": ["bhau", "lagech", "paisa", "confirm kara"],
        "ta": ["thambi", "udane", "panam"],
        "en": ["bro", "urgent", "money"],
    }
    add = rng.choice(fillers.get(lang, ["ok"]))
    return text + " " + add


def _noise(text: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        text = text.replace(" ", "  ", 1)
    if rng.random() < 0.3:
        text = text + " !!!"
    if rng.random() < 0.2:
        text = text.replace("₹", "Rs ")
    return text


def fill(text: str, rng: random.Random) -> str:
    """Fill ``{slot}`` placeholders from SLOTS."""
    def sub(m: "re.Match[str]") -> str:
        key = m.group(1)
        vals = SLOTS.get(key)
        return rng.choice(vals) if vals else m.group(0)

    return re.sub(r"\{(\w+)\}", sub, text)


# --------------------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------------------
def split_of(template_id: str) -> str:
    """Stable, template-level split. 70% train / 15% val / 15% test, by hash of the template id."""
    h = int(hashlib.sha256(template_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


# --------------------------------------------------------------------------------------
# row construction
# --------------------------------------------------------------------------------------
def _soft_distribution(p_true: float, soft: bool, rng: random.Random, k: int = 2) -> List[float]:
    """Target distribution for a question, near-one-hot unless the template is ambiguous."""
    if k == 2:
        if soft:
            # centred on the template's own probability, jittered a little
            p = min(0.95, max(0.05, p_true + rng.uniform(-0.12, 0.12)))
            return [1.0 - p, p]
        p = max(p_true, 0.94) if p_true >= 0.5 else min(p_true, 0.06)
        return [1.0 - p, p]
    return [1.0 / k] * k


def build_rows(tasks: Dict[str, Any], templates: Sequence[Template], n_variants: int = 8,
               seed: int = 7, romanized: bool = True) -> List[Dict[str, Any]]:
    """Expand templates into training/eval rows.

    Each row: ``{id, task, split, template, lang, state, questions, gold, option_orders}`` where
    ``questions`` uses the *runtime* shape and ``gold`` carries ``{"probabilities": {...}}``
    exactly like the official Laya fine-tuning dataset, so the same collate/convert code path
    works for both.
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for tpl in templates:
        spec = tasks[tpl.task]
        langs = list(tpl.text.keys())
        if romanized:
            langs = langs + [f"{lg}-rom" for lg in tpl.text if lg in ROMANIZED]
        for lang in langs:
            base_lang = lang.split("-")[0]
            for v in range(n_variants):
                state = fill(tpl.text[base_lang], rng)
                if lang.endswith("-rom"):
                    state = _romanize_hints(state, rng, base_lang)
                if rng.random() < 0.35:
                    state = rng.choice(_PREFIX).format(city=rng.choice(SLOTS["city"])) + " " + state
                if rng.random() < 0.3:
                    state = state + rng.choice(_SUFFIX)
                if rng.random() < 0.22:
                    state = _obfuscate(state, rng)
                if rng.random() < 0.25:
                    state = _noise(state, rng)

                questions: Dict[str, Any] = {}
                gold: Dict[str, Any] = {}
                orders: Dict[str, List[int]] = {}

                if tpl.triage and spec.get("triage"):
                    tri_keys = list(spec["triage"]["criteria"].keys())
                    if tpl.triage not in tri_keys:
                        raise ValueError(f"{tpl.id}: triage label {tpl.triage!r} not in claim_v1 triage criteria")
                    k = len(tri_keys)
                    probs = [0.02 / max(1, k - 1)] * k
                    probs[tri_keys.index(tpl.triage)] = 0.98 if not tpl.soft else 0.72
                    if tpl.soft:
                        # spread the remainder the way an unsure model would
                        rest = (1.0 - probs[tri_keys.index(tpl.triage)]) / max(1, k - 1)
                        probs = [rest] * k
                        probs[tri_keys.index(tpl.triage)] = 0.72
                    s = sum(probs)
                    probs = [p / s for p in probs]
                    questions["triage"] = spec["triage"]
                    gold["triage"] = {"probabilities": {key: round(pr, 4) for key, pr in zip(tri_keys, probs)}}
                    order = list(range(k))
                    if rng.random() < 0.5:  # option-order augmentation
                        rng.shuffle(order)
                    orders["triage"] = order

                    fups = spec.get("followups", {}).get(tpl.triage, {})
                    for qid, q in fups.items():
                        p_true = float(tpl.followups.get(qid, 0.0))
                        dist = _soft_distribution(p_true, tpl.soft, rng)
                        questions[qid] = q
                        gold[qid] = {"probabilities": {"false": round(dist[0], 4), "true": round(dist[1], 4)}}
                        order = [0, 1]
                        if rng.random() < 0.5:
                            order = [1, 0]
                        orders[qid] = order

                rows.append({
                    "id": f"{tpl.id}.{lang}.{v}",
                    "task": tpl.task,
                    "split": split_of(tpl.id),
                    "template": tpl.id,
                    "lang": lang,
                    "state": state,
                    "questions": questions,
                    "gold": gold,
                    "option_orders": orders,
                    "negative": tpl.negative,
                })
    rng.shuffle(rows)
    return rows


def load_tasks(banks_path: str) -> Dict[str, Any]:
    with open(banks_path, "r", encoding="utf-8") as f:
        return json.load(f)["tasks"]


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dataset_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, int] = {}
    by_task: Dict[str, int] = {}
    by_lang: Dict[str, int] = {}
    templates: Dict[str, set] = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + 1
        templates.setdefault(r["split"], set()).add(r["template"])
    return {
        "rows": len(rows),
        "by_split": by_split,
        "by_task": by_task,
        "by_lang": by_lang,
        "templates": {k: len(v) for k, v in templates.items()},
        "template_overlap": _overlap(templates),
    }


def _overlap(templates: Dict[str, set]) -> Dict[str, int]:
    """How many templates appear in more than one split. Must be 0 — this is the leakage gate."""
    out: Dict[str, int] = {}
    keys = sorted(templates)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out[f"{a}&{b}"] = len(templates[a] & templates[b])
    return out
