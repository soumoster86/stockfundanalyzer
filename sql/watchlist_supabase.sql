-- Watchlist storage for Fundamental Stock Analyzer (Supabase / Postgres)
-- Run this in the Supabase SQL editor once.

create table if not exists public.watchlist_items (
  id bigint generated always as identity primary key,
  username text not null,
  ticker text not null,
  created_at timestamptz not null default now(),
  unique (username, ticker)
);

create index if not exists watchlist_items_username_idx
  on public.watchlist_items (username);

comment on table public.watchlist_items is
  'Per-user stock watchlist for the Streamlit Fundamental Stock Analyzer';

-- Optional: enable RLS and allow only service role from the Streamlit server.
-- For a private Streamlit app that uses the service_role key, RLS can stay off
-- or use a simple policy. Example (if using anon key + custom JWT later):
--
-- alter table public.watchlist_items enable row level security;
-- create policy "service full access" on public.watchlist_items
--   for all using (true) with check (true);
