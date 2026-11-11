import { afterEach, expect, test } from 'bun:test'

import { apiKey, keyWanted, setApiKey } from '../lib/apikey'
import { api, dispositionName, query, readHealth } from './client'

const real = globalThis.fetch
const calls: { url: string; headers: Record<string, string> }[] = []

/** Answers every request with `respond`, keeping what was asked. */
function serve(respond: (url: string) => Response) {
    calls.length = 0
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        calls.push({ url, headers: (init?.headers ?? {}) as Record<string, string> })
        return respond(url)
    }) as typeof fetch
}

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

afterEach(() => {
    globalThis.fetch = real
    setApiKey(undefined)
})

test('the API key goes in the X-API-Key header of every request, never in the URL', async () => {
    serve(() => json([]))
    setApiKey('  s3cret ')
    expect(apiKey()).toBe('s3cret')
    await api.jobs()
    await api.datasetData('01J', { locations: ['A B', 'C'] })
    serve(() => new Response('# metrics\n'))
    await api.metrics()
    for (const call of calls) {
        expect(call.headers['x-api-key']).toBe('s3cret')
        expect(call.url).not.toContain('s3cret')
    }
})

test('a 401 asks for a key', async () => {
    serve(() => json({ title: 'Unauthorized', detail: 'Missing API key' }, 401))
    expect(keyWanted()).toBe(false)
    await expect(api.info()).rejects.toMatchObject({ status: 401, detail: 'Missing API key' })
    expect(keyWanted()).toBe(true)
    setApiKey('given')
    expect(keyWanted()).toBe(false)
})

test('an unhealthy service answers 503 with its health, which is kept', async () => {
    serve(() => json({ status: 'unhealthy', checks: { database: { state: 'unhealthy', message: 'Check failed: locked' } } }, 503))
    expect(await readHealth()).toEqual({ status: 'unhealthy', checks: { database: { state: 'unhealthy', message: 'Check failed: locked' } } })
    serve(() => json({ title: 'Service Unavailable' }, 503))
    await expect(readHealth()).rejects.toMatchObject({ status: 503 })
})

test('a dataset read asks the service for canonical periods and its projection', async () => {
    serve(() => json({ columns: [], data: [] }))
    await api.datasetData('01J', { locations: ['A&B', 'C'], columns: ['cases'], before: '2024-9', last: 12 })
    expect(calls[0]?.url).toBe('/api/v1/datasets/01J/$data?location=A%26B&location=C&column=cases&before=2024-9&last=12&canonical=true')
})

test('past results are read a page at a time, newest first', async () => {
    serve(() => json({ items: [], total: 0, page: 2, size: 25, pages: 0 }))
    await api.evaluations(2, 25)
    expect(calls[0]?.url).toBe('/api/v1/artifacts?type=ml_backtest&order=desc&page=2&size=25')
})

test('query strings and download names', () => {
    expect(query({})).toBe('')
    expect(query({ type: undefined, page: 1 })).toBe('?page=1')
    expect(dispositionName('attachment; filename="evaluation.json"')).toBe('evaluation.json')
    expect(dispositionName("attachment; filename*=UTF-8''%C3%B8st.json")).toBe('øst.json')
    expect(dispositionName(null)).toBeUndefined()
})
