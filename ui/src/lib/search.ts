import type { SearchGroup, SearchResult } from '@winterop-com/sokkel-ui/lib/search'
import { Database, Gauge, SlidersHorizontal, TrendingUp } from 'lucide-react'

import { api } from '../api/client'
import type { Artifact, Config, Dataset, EvaluationSummary, ForecastSummary, Page } from '../api/types'
import { count, periodWord, score, when } from './format'

/** How many of each kind a search lists; the heading says how many match in all. */
export const SEARCH_SIZE = 5

/** What the service found for one search, a page of each kind. */
export interface Found {
    datasets: Page<Dataset>
    evaluations: Page<Artifact<never, EvaluationSummary>>
    forecasts: Page<Artifact<never, ForecastSummary>>
    configs: Page<Config>
}

/** The kinds a search finds, in the order they are listed. */
export type SearchKind = 'datasets' | 'evaluations' | 'forecasts' | 'configs'

/** The route a found record opens at. */
export function resultRoute(kind: string, id: string): string {
    if (kind === 'datasets') return `/datasets/${id}`
    if (kind === 'evaluations') return `/evaluate/${id}`
    if (kind === 'forecasts') return `/forecast/${id}`
    return `/configs/${id}`
}

/** A record made from a dataset, named for it: its name when known, else its id; request data without one. */
const madeFrom = (id: string | null | undefined, names: ReadonlyMap<string, string>) =>
    id === null || id === undefined ? 'Request data' : (names.get(id) ?? id)

/** A config's options in one line. */
const options = (data: Record<string, unknown>) =>
    Object.entries(data)
        .map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`)
        .join('  ')

/**
 * What a search found, grouped by kind for `Search` and the palette, each record named for a
 * reader: a dataset or config by its name, an evaluation or forecast by its dataset's name from
 * `names`. Kinds with nothing found are left out.
 */
export function searchGroups(found: Found, names: ReadonlyMap<string, string>): SearchGroup[] {
    const group = (kind: SearchKind, title: string, page: Page<unknown>, results: SearchResult[]): SearchGroup => ({
        kind,
        title,
        results,
        total: page.total,
    })
    return [
        group(
            'datasets',
            'Datasets',
            found.datasets,
            found.datasets.items.map((dataset) => ({
                id: dataset.id,
                title: dataset.name,
                icon: Database,
                detail: `${periodWord(dataset.period_type)}, ${dataset.first_period ?? '-'} to ${dataset.last_period ?? '-'}, ${count(dataset.locations)} locations`,
                meta: `${count(dataset.rows)} rows`,
            })),
        ),
        group(
            'evaluations',
            'Evaluations',
            found.evaluations,
            found.evaluations.items.map((run) => {
                const summary = run.data.summary
                const setup = summary?.options ? `${summary.options.splits ?? '-'} splits x ${summary.options.periods ?? '-'} periods, ` : ''
                return {
                    id: run.id,
                    title: madeFrom(summary?.dataset_id, names),
                    icon: Gauge,
                    detail: `${setup}${when(run.created_at)}`,
                    meta: `CRPS ${score(summary?.aggregate.crps)}`,
                }
            }),
        ),
        group(
            'forecasts',
            'Forecasts',
            found.forecasts,
            found.forecasts.items.map((run) => {
                const summary = run.data.summary
                return {
                    id: run.id,
                    title: madeFrom(summary?.dataset_id, names),
                    icon: TrendingUp,
                    detail: `${summary?.periods ?? '-'} periods from ${summary?.first_period ?? '-'}, ${when(run.created_at)}`,
                    meta: summary?.locations === undefined ? undefined : `${count(summary.locations)} locations`,
                }
            }),
        ),
        group(
            'configs',
            'Configs',
            found.configs,
            found.configs.items.map((config) => ({
                id: config.id,
                title: config.name,
                icon: SlidersHorizontal,
                detail: options(config.data) || 'Model defaults',
                meta: when(config.created_at),
            })),
        ),
    ].filter((each) => each.results.length > 0)
}

/** The datasets the evaluations and forecasts found were made from that `names` does not name yet. */
export function unnamed(found: Found, names: ReadonlyMap<string, string>): string[] {
    const ids = [...found.evaluations.items, ...found.forecasts.items].map((run) => run.data.summary?.dataset_id)
    return [...new Set(ids)].filter((id): id is string => typeof id === 'string' && !names.has(id))
}

/** Dataset names by id, kept between searches: a dataset is never renamed. */
const NAMES = new Map<string, string>()

/**
 * Searches the service for `query`: the datasets and configs whose name holds it, and the
 * evaluations and forecasts made from such a dataset, a few of each, newest results first for runs.
 */
export async function searchService(query: string, signal: AbortSignal): Promise<SearchGroup[]> {
    const q = query.trim()
    if (q === '') return []
    const [datasets, evaluations, forecasts, configs] = await Promise.all([
        api.findDatasets(q, SEARCH_SIZE, signal),
        api.evaluations(1, SEARCH_SIZE, signal, q),
        api.forecasts(1, SEARCH_SIZE, signal, q),
        api.findConfigs(q, SEARCH_SIZE, signal),
    ])
    const found: Found = { datasets, evaluations, forecasts, configs }
    for (const dataset of datasets.items) NAMES.set(dataset.id, dataset.name)
    // A run's dataset matched too, but may be past the few datasets listed: read its name.
    await Promise.all(
        unnamed(found, NAMES).map((id) =>
            api.dataset(id, signal).then(
                (dataset) => NAMES.set(id, dataset.name),
                () => undefined,
            ),
        ),
    )
    return searchGroups(found, NAMES)
}
