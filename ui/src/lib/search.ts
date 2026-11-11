import type { SearchGroup, SearchResult } from '@winterop-com/sokkel-ui/lib/search'
import { Database, Gauge, SlidersHorizontal, TrendingUp } from 'lucide-react'

import { api } from '../api/client'
import type { Artifact, Config, Dataset, EvaluationSummary, ForecastSummary, Page } from '../api/types'
import { count, periodWord, score, when } from './format'

/** How many of each kind a search lists; the heading says how many match in all. */
export const SEARCH_SIZE = 5

/** How many of each kind a page of the full results adds. */
export const RESULTS_SIZE = 25

/** What the service found for one search, a page of each kind. */
export interface Found {
    datasets: Page<Dataset>
    evaluations: Page<Artifact<never, EvaluationSummary>>
    forecasts: Page<Artifact<never, ForecastSummary>>
    configs: Page<Config>
}

/** The kinds a search finds, in the order they are listed. */
export type SearchKind = 'datasets' | 'evaluations' | 'forecasts' | 'configs'

export const SEARCH_KINDS: readonly SearchKind[] = ['datasets', 'evaluations', 'forecasts', 'configs']

/** The page of each kind's matches to read, by kind; a kind not named is not read. */
export type KindPages = Partial<Record<SearchKind, number>>

/** The route a found record opens at. */
export function resultRoute(kind: string, id: string): string {
    if (kind === 'datasets') return `/datasets/${id}`
    if (kind === 'evaluations') return `/evaluate/${id}`
    if (kind === 'forecasts') return `/forecast/${id}`
    return `/configs/${id}`
}

/** Says a search failed, with the reason the service gave, in the top bar's search and the palette. */
export const searchFailed = (reason: string): string => `The search did not answer: ${reason}`

/** The address of the full results for a query, which keeps the query. */
export const searchRoute = (query: string): string => `/search?q=${encodeURIComponent(query.trim())}`

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
export function searchGroups(found: Partial<Found>, names: ReadonlyMap<string, string>): SearchGroup[] {
    const none: Page<never> = { items: [], total: 0, page: 1, size: 0, pages: 0 }
    const { datasets = none, evaluations = none, forecasts = none, configs = none } = found
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
            datasets,
            datasets.items.map((dataset) => ({
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
            evaluations,
            evaluations.items.map((run) => {
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
            forecasts,
            forecasts.items.map((run) => {
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
            configs,
            configs.items.map((config) => ({
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
export function unnamed(found: Partial<Found>, names: ReadonlyMap<string, string>): string[] {
    const ids = [...(found.evaluations?.items ?? []), ...(found.forecasts?.items ?? [])].map((run) => run.data.summary?.dataset_id)
    return [...new Set(ids)].filter((id): id is string => typeof id === 'string' && !names.has(id))
}

/** Dataset names by id, kept between searches: a dataset is never renamed. */
const NAMES = new Map<string, string>()

/** Reads the asked page of each kind's matches for `q`, newest first for runs. */
const FIND: { [K in SearchKind]: (q: string, page: number, size: number, signal: AbortSignal) => Promise<Found[K]> } = {
    datasets: (q, page, size, signal) => api.findDatasets(q, page, size, signal),
    evaluations: (q, page, size, signal) => api.evaluations(page, size, signal, q),
    forecasts: (q, page, size, signal) => api.forecasts(page, size, signal, q),
    configs: (q, page, size, signal) => api.findConfigs(q, page, size, signal),
}

/**
 * Asks the service for the given page of each kind named in `pages` whose name holds `query` (an
 * evaluation or forecast by its dataset's name), and reads the names of the runs' datasets not yet
 * known.
 */
export async function findPages(query: string, pages: KindPages, size: number, signal: AbortSignal): Promise<Partial<Found>> {
    const q = query.trim()
    const kinds = SEARCH_KINDS.filter((kind) => pages[kind] !== undefined)
    const read = await Promise.all(kinds.map((kind) => FIND[kind](q, pages[kind]!, size, signal)))
    const found: Partial<Found> = Object.fromEntries(kinds.map((kind, index) => [kind, read[index]]))
    for (const dataset of found.datasets?.items ?? []) NAMES.set(dataset.id, dataset.name)
    // A run's dataset matched too, but may be past the datasets listed: read its name.
    await Promise.all(
        unnamed(found, NAMES).map((id) =>
            api.dataset(id, signal).then(
                (dataset) => NAMES.set(id, dataset.name),
                () => undefined,
            ),
        ),
    )
    return found
}

/** What two reads found together: each kind's items in order, its total from the later read. */
export function joinFound(before: Partial<Found>, after: Partial<Found>): Partial<Found> {
    const joined: Record<string, Page<unknown>> = { ...before }
    for (const kind of SEARCH_KINDS) {
        const later = after[kind] as Page<unknown> | undefined
        if (later === undefined) continue
        const earlier = joined[kind]
        joined[kind] = earlier === undefined ? later : { ...later, items: [...earlier.items, ...later.items] }
    }
    return joined as Partial<Found>
}

/** The next page of each kind more of which match than `found` holds; empty when it holds them all. */
export function nextPages(found: Partial<Found>, size: number, kinds: readonly SearchKind[] = SEARCH_KINDS): KindPages {
    const pages: KindPages = {}
    for (const kind of kinds) {
        const page = found[kind]
        if (page !== undefined && page.items.length < page.total) pages[kind] = Math.floor(page.items.length / size) + 1
    }
    return pages
}

/** Groups what was found for `Search`, `SearchResults` and the palette, each record named. */
export const foundGroups = (found: Partial<Found>): SearchGroup[] => searchGroups(found, NAMES)

/** The first page of every kind. */
const FIRST: KindPages = { datasets: 1, evaluations: 1, forecasts: 1, configs: 1 }

/**
 * Searches the service for `query`: the datasets and configs whose name holds it, and the
 * evaluations and forecasts made from such a dataset, a few of each, newest results first for runs.
 */
export async function searchService(query: string, signal: AbortSignal): Promise<SearchGroup[]> {
    if (query.trim() === '') return []
    return foundGroups(await findPages(query, FIRST, SEARCH_SIZE, signal))
}
