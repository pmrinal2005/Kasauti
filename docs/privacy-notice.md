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
There is no browser on a feature-phone call, so the telephony channel cannot do speech-to-text on the
device. It is served by `/api/ivr`, and the rules are:

- it is **off unless `IVR_ENABLED=1`**, and `/api/health` reports `audioRoutes: 1` only then. A normal
  deploy answers `audioRoutes: 0`;
- your telephony provider (Exotel, Twilio, Plivo…) records and transcribes the call under **its**
  policy — that is unavoidable on a phone line, and it is why this channel is a separate, documented
  exception rather than something we quietly do;
- Kasauti receives only the transcribed text, runs the same deterministic on-device checker on the
  server for that one call, and **stores neither the audio nor the transcript**;
- the answer is a fixed spoken line for the verdict, in English or Hindi. No language model is on the
  phone line, so nothing can be invented on it;
- what is kept is the same record the website keeps for a cache hit: a SHA-256 of the normalised text
  plus evidence codes — no text, no phone number, no call id (caller ids are hashed and only used as
  an in-memory rate-limit key).
- the phone path is **unverified against a live provider** as of this build. Treat it as ready to
  wire up, not as a working helpline.

## What Kasauti never does
- No `/api/asr` route exists; no audio upload code path exists on the web surface.
- No trackers, no SMS/contacts permissions, no account required.
- Your check history and on-device verdict cache live only in this browser's local storage; "Erase local history" deletes them.
- Kasauti never recommends a stock, fund, broker or trade.

Fraud? Call **1930** or report at **cybercrime.gov.in**.
