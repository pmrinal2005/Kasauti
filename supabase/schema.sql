-- Kasauti shared verdict cache (optional). Stores hashes + evidence codes ONLY — never raw message text or audio.
create table if not exists public.verdicts (
  hash          text primary key check (hash ~ '^[0-9a-f]{64}$'),
  simhash       text not null check (simhash ~ '^[0-9a-f]{16}$'),
  rung          text not null check (rung in ('safe_pattern','unclear','caution','likely_scam','known_scam')),
  codes         text[] not null default '{}',
  model_version text not null default '',
  hits          integer not null default 1,
  updated_at    timestamptz not null default now()
);
create index if not exists verdicts_simhash_idx on public.verdicts (simhash);
create index if not exists verdicts_updated_idx on public.verdicts (updated_at desc);
-- Only the server (service-role key, Vercel env) writes; no anon access.
alter table public.verdicts enable row level security;
