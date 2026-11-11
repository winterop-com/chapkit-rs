import { afterEach, expect, test } from 'bun:test'

import type { Artifact, Config, Dataset, EvaluationSummary, ForecastSummary, Page } from '../api/types'
import { resultRoute, searchGroups, searchService, unnamed, type Found } from './search'

const page = <T>(items: T[], total = items.length): Page<T> => ({ items, total, page: 1, size: 5, pages: 1 })

const dataset = (id: string, name: string) =>
    ({ id, name, created_at: '2026-01-01T00:00:00', updated_at: '', rows: 1200, locations: 3, location_ids: [], period_type: 'month', first_period: '2010-01', last_period: '2012-12', periods: 36, columns: [], covariates: [] }) as Dataset

const evaluation = (id: string, datasetId: string | null) =>
    ({ id, level: 0, tags: [], created_at: '2026-01-02T00:00:00', updated_at: '', data: { type: 'ml_backtest', summary: { dataset_id: datasetId, data: {}, options: { splits: 7, periods: 3 }, aggregate: { crps: 1.2345 } } } }) as Artifact<never, EvaluationSummary>

const forecast = (id: string, datasetId: string) =>
    ({ id, level: 0, tags: [], created_at: '2026-01-03T00:00:00', updated_at: '', data: { type: 'ml_forecast', summary: { dataset_id: datasetId, periods: 6, first_period: '2013-01', locations: 3 } } }) as Artifact<never, ForecastSummary>

const config = (id: string, name: string) => ({ id, name, data: { alpha: 0.5 }, tags: [], created_at: '2026-01-04T00:00:00', updated_at: '' }) as Config

const found = (): Found => ({
    datasets: page([dataset('D1', 'laos monthly')], 12),
    evaluations: page([evaluation('E1', 'D1'), evaluation('E2', 'D9'), evaluation('E3', null)]),
    forecasts: page([forecast('F1', 'D9')]),
    configs: page<Config>([]),
})

test('found records are grouped by kind, named for a reader, and empty kinds left out', () => {
    const groups = searchGroups(found(), new Map([['D1', 'laos monthly'], ['D9', 'laos weekly']]))
    expect(groups.map((group) => [group.kind, group.title, group.total, group.results.length])).toEqual([
        ['datasets', 'Datasets', 12, 1],
        ['evaluations', 'Evaluations', 3, 3],
        ['forecasts', 'Forecasts', 1, 1],
    ])
    const [datasets, evaluations, forecasts] = groups
    expect(datasets!.results[0]).toMatchObject({ id: 'D1', title: 'laos monthly', detail: 'Monthly, 2010-01 to 2012-12, 3 locations', meta: '1,200 rows' })
    expect(evaluations!.results.map((result) => result.title)).toEqual(['laos monthly', 'laos weekly', 'Request data'])
    expect(evaluations!.results[0]).toMatchObject({ meta: 'CRPS 1.23' })
    expect(evaluations!.results[0]!.detail).toStartWith('7 splits x 3 periods, ')
    expect(forecasts!.results[0]).toMatchObject({ title: 'laos weekly', meta: '3 locations' })
    expect(forecasts!.results[0]!.detail).toStartWith('6 periods from 2013-01, ')
})

test('a run whose dataset has no known name is called by its id, and asked for', () => {
    const names = new Map([['D1', 'laos monthly']])
    expect(unnamed(found(), names)).toEqual(['D9'])
    expect(searchGroups(found(), names)[1]!.results[1]!.title).toBe('D9')
})

test('configs are found by name with their options as the line under', () => {
    const groups = searchGroups({ ...found(), configs: page([config('C1', 'fast')]) }, new Map())
    expect(groups.at(-1)).toMatchObject({ kind: 'configs', results: [{ id: 'C1', title: 'fast', detail: 'alpha=0.5' }] })
})

test('each kind opens at its own page', () => {
    expect(['datasets', 'evaluations', 'forecasts', 'configs'].map((kind) => resultRoute(kind, 'X'))).toEqual([
        '/datasets/X',
        '/evaluate/X',
        '/forecast/X',
        '/configs/X',
    ])
})

const real = globalThis.fetch
afterEach(() => {
    globalThis.fetch = real
})

test('a search asks the service for a few of each kind by name, and the names of the runs\' datasets', async () => {
    const urls: string[] = []
    globalThis.fetch = (async (input: RequestInfo | URL) => {
        const url = String(input)
        urls.push(url)
        const body = url.includes('/datasets/D9')
            ? dataset('D9', 'laos weekly')
            : url.includes('/datasets?')
              ? page([dataset('D1', 'laos monthly')])
              : url.includes('ml_backtest')
                ? page([evaluation('E2', 'D9')])
                : page([])
        return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } })
    }) as typeof fetch
    expect(await searchService('   ', new AbortController().signal)).toEqual([])
    expect(urls).toEqual([])
    const groups = await searchService(' laos ', new AbortController().signal)
    expect(urls.slice(0, 4)).toEqual([
        '/api/v1/datasets?q=laos&page=1&size=5',
        '/api/v1/artifacts?type=ml_backtest&order=desc&page=1&size=5&q=laos',
        '/api/v1/artifacts?type=ml_forecast&order=desc&page=1&size=5&q=laos',
        '/api/v1/configs?q=laos&page=1&size=5',
    ])
    expect(urls[4]).toBe('/api/v1/datasets/D9')
    expect(groups[1]!.results[0]!.title).toBe('laos weekly')
})
