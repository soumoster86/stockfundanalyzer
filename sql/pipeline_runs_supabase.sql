-- Pipeline run log for Fundamental Stock Analyzer (Supabase / Postgres)
-- Run once in the Supabase SQL editor (same project as watchlists is fine).

create table if not exists public.pipeline_runs (
  id bigint generated always as identity primary key,
  workflow_run text,
  repository text,
  ref text,
  status text not null default 'success',
  source text not null default 'github-actions daily-fundamentals',
  started_at timestamptz,
  finished_at timestamptz not null default now(),
  n_tickers integer,
  n_rows integer,
  max_tickers integer,
  use_sector boolean,
  avg_quality double precision,
  median_quality double precision,
  n_data_warnings integer,
  commit_csv boolean,
  workflow_url text,
  error_message text,
  details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

-- One row per GitHub Actions run id (Postgres allows multiple NULLs)
create unique index if not exists pipeline_runs_workflow_run_uidx
  on public.pipeline_runs (workflow_run);

create index if not exists pipeline_runs_finished_at_idx
  on public.pipeline_runs (finished_at desc);

comment on table public.pipeline_runs is
  'GitHub daily fundamentals / rescore pipeline history for the Streamlit app';
