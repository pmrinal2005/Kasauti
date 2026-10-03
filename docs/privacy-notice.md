# Kasauti — Voice & Data Privacy Notice

**Short version:** Kasauti's servers never receive, store or transmit a single byte of your voice.

| Data | Processed by | Leaves your device? |
|---|---|---|
| Message text (Tier-0 checks) | Your device (Web Worker) | Never |
| Laya classification (Tier-1) | Your device (ONNX model in a Web Worker) | Never |
| Voice → text | Your browser's own speech engine (`SpeechRecognition`) | Kasauti never receives it. If your browser cannot recognise speech on-device (`processLocally`), the browser vendor (Google for Chrome, Apple for Safari) processes it under *their* policy. The app always shows which mode is active. |
| Text → voice | Your device (`speechSynthesis`) | Never |
| Verdict fingerprint | `/api/v/{hash}` shared cache | Only a SHA-256 hash, a SimHash and evidence codes — no message text |
| "Why?" explanation | Groq (text only) | Only the text you chose to explain, only when you tap "Why?" |
| Model download | Hugging Face CDN (public model repo) | A normal file download; no message data is sent |

## The one documented exception: phone calls (IVR)
There is no browser on a feature-phone call, so the telephony channel (when sponsored) uses the
telephony provider's transcription. That channel is **not built yet**; when it is, it will live in an
isolated `server/telephony/` module, will never be used by the website or app, and will be disclosed here.

## What Kasauti never does
- No `/api/asr` route exists; no audio upload code path exists on the web surface.
- No trackers, no SMS/contacts permissions, no account required.
- Your check history and on-device verdict cache live only in this browser's local storage; "Erase local history" deletes them.
- Kasauti never recommends a stock, fund, broker or trade.

Fraud? Call **1930** or report at **cybercrime.gov.in**.
