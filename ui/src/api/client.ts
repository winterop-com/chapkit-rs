import { apiKey, keyRefused } from '../lib/apikey'
import type {
    Artifact,
    ArtifactNode,
    BacktestOptions,
    Config,
    Dataset,
    Evaluation,
    EvaluationSummary,
    Forecast,
    ForecastSummary,
    Frame,
    Health,
    Info,
    Job,
    JobSubmitted,
    Page,
    ProblemBody,
    SystemInfo,
} from './types'

/** A request the service refused or could not answer, with its problem+json said for a person. */
export class ApiError extends Error {
    readonly status: number | 'unreachable'
    readonly detail: string | undefined

    constructor(status: number | 'unreachable', title: string, detail?: string) {
        super(title)
        this.status = status
        this.detail = detail
    }
}

/** Anything a read threw, as an ApiError. */
export const asApiError = (failure: unknown): ApiError =>
    failure instanceof ApiError ? failure : new ApiError(0, 'Something went wrong', String(failure))

/** A read stopped because nothing waits for it any more: never shown. */
export const aborted = (failure: unknown): boolean => failure instanceof DOMException && failure.name === 'AbortError'

/** Sends a request with the API key, if one is given; a 401 asks the person for one. */
async function send(path: string, init?: RequestInit, accept = 'application/json'): Promise<Response> {
    const key = apiKey()
    let response: Response
    try {
        response = await fetch(path, {
            ...init,
            headers: { accept, ...(key === undefined ? {} : { 'x-api-key': key }), ...init?.headers },
        })
    } catch (error) {
        if (aborted(error)) throw error
        throw new ApiError('unreachable', 'The service does not answer', String(error))
    }
    if (response.status === 401) keyRefused()
    return response
}

/** The error a refused response says. */
async function refusal(response: Response): Promise<ApiError> {
    let body: ProblemBody = {}
    try {
        body = (await response.json()) as ProblemBody
    } catch {
        // Not problem+json: the status says it.
    }
    return new ApiError(response.status, body.title ?? response.statusText, body.detail ?? undefined)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await send(path, init)
    if (!response.ok) throw await refusal(response)
    if (response.status === 204) return undefined as T
    return (await response.json()) as T
}

/** `/health`: a health document on 200 and on 503 (unhealthy), which is an answer, not a failure. */
export async function readHealth(signal?: AbortSignal): Promise<Health> {
    const response = await send('/health', { signal })
    if (response.ok || response.status === 503) {
        try {
            const body = (await response.json()) as Partial<Health>
            if (body.status === 'healthy' || body.status === 'degraded' || body.status === 'unhealthy') return body as Health
        } catch {
            // Not a health document: refused below.
        }
    }
    throw await refusal(response)
}

const json = (method: string, body: unknown): RequestInit => ({
    method,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
})

/** A list endpoint answers a list, or a page of one when paged. */
function items<T>(list: T[] | { items: T[] }): T[] {
    return Array.isArray(list) ? list : list.items
}

/** A query string from its parameters, repeating a list's; empty when there are none. */
export function query(params: Record<string, string | number | boolean | readonly string[] | undefined>): string {
    const search = new URLSearchParams()
    for (const [name, value] of Object.entries(params)) {
        if (value === undefined) continue
        for (const each of typeof value === 'object' ? value : [value]) search.append(name, String(each))
    }
    const text = search.toString()
    return text === '' ? '' : `?${text}`
}

/** The file name a `Content-Disposition` header gives, if any. */
export function dispositionName(header: string | null): string | undefined {
    if (header === null) return undefined
    const encoded = /filename\*=(?:UTF-8'')?([^;]+)/i.exec(header)
    if (encoded?.[1] !== undefined) {
        try {
            return decodeURIComponent(encoded[1].trim().replace(/^"|"$/g, ''))
        } catch {
            // Not percent-encoded as it should be: the plain name below.
        }
    }
    return /filename="?([^";]+)"?/i.exec(header)?.[1]?.trim()
}

/** Hands the reader a file. */
export function saveBlob(name: string, blob: Blob): void {
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = name
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
}

const BASE = '/api/v1'

export interface BacktestRequest extends BacktestOptions {
    dataset_id: string
    config_id?: string
}

export interface ForecastRequest {
    dataset_id: string
    config_id?: string
    periods?: number
    weather?: string
}

/** What of a stored dataset to read: see `GET /api/v1/datasets/{id}/$data`. */
export interface DataProjection {
    locations?: readonly string[]
    columns?: readonly string[]
    /** Only the periods before this one. */
    before?: string
    /** Only the last this many periods. */
    last?: number
}

type Signal = AbortSignal | undefined

export const api = {
    info: (signal?: Signal) => request<Info>(`${BASE}/info`, { signal }),
    system: (signal?: Signal) => request<SystemInfo>(`${BASE}/system`, { signal }),
    health: readHealth,
    openapi: (signal?: Signal) =>
        request<{ paths: Record<string, Record<string, { summary?: string; tags?: string[] }>> }>('/openapi.json', { signal }),
    /** The Prometheus text of `/metrics`. */
    metrics: async (signal?: Signal) => {
        const response = await send('/metrics', { signal }, 'text/plain')
        if (!response.ok) throw await refusal(response)
        return response.text()
    },

    configs: async (signal?: Signal) => items(await request<Config[] | { items: Config[] }>(`${BASE}/configs`, { signal })),
    /** A page of the configs whose name holds `q`. */
    findConfigs: (q: string, size: number, signal?: Signal) => request<Page<Config>>(`${BASE}/configs${query({ q, page: 1, size })}`, { signal }),
    configSchema: (signal?: Signal) => request<Record<string, unknown>>(`${BASE}/configs/$schema`, { signal }),
    createConfig: (name: string, data: unknown) => request<Config>(`${BASE}/configs`, json('POST', { name, data })),
    deleteConfig: (id: string) => request<void>(`${BASE}/configs/${id}`, { method: 'DELETE' }),

    jobs: (signal?: Signal) => request<Job[]>(`${BASE}/jobs`, { signal }),
    job: (id: string, signal?: Signal) => request<Job>(`${BASE}/jobs/${id}`, { signal }),

    /** The artifacts, of one type when given; a list carries no content, which `artifact` reads. */
    artifacts: async (type?: string, signal?: Signal) =>
        items(await request<Artifact[] | { items: Artifact[] }>(`${BASE}/artifacts${query({ type })}`, { signal })),
    /**
     * A page of the newest artifacts of a type, each with its summary in place of its content; with
     * `q`, only those made from a dataset whose name holds it.
     */
    results: <S>(type: string, page: number, size: number, signal?: Signal, q?: string) =>
        request<Page<Artifact<never, S>>>(`${BASE}/artifacts${query({ type, order: 'desc', page, size, q })}`, { signal }),
    evaluations: (page: number, size: number, signal?: Signal, q?: string) => api.results<EvaluationSummary>('ml_backtest', page, size, signal, q),
    forecasts: (page: number, size: number, signal?: Signal, q?: string) => api.results<ForecastSummary>('ml_forecast', page, size, signal, q),
    artifact: <C = unknown>(id: string, signal?: Signal) => request<Artifact<C>>(`${BASE}/artifacts/${id}`, { signal }),
    artifactTree: (id: string, signal?: Signal) => request<ArtifactNode>(`${BASE}/artifacts/${id}/$tree`, { signal }),
    /** Downloads an artifact's content as a file, sent with the key like every other request. */
    downloadArtifact: async (id: string) => {
        const response = await send(`${BASE}/artifacts/${id}/$download`, undefined, '*/*')
        if (!response.ok) throw await refusal(response)
        saveBlob(dispositionName(response.headers.get('content-disposition')) ?? `artifact-${id}`, await response.blob())
    },
    deleteArtifact: (id: string) => request<void>(`${BASE}/artifacts/${id}`, { method: 'DELETE' }),

    datasets: async (signal?: Signal) => items(await request<Dataset[] | { items: Dataset[] }>(`${BASE}/datasets`, { signal })),
    /** A page of the datasets whose name holds `q`. */
    findDatasets: (q: string, size: number, signal?: Signal) => request<Page<Dataset>>(`${BASE}/datasets${query({ q, page: 1, size })}`, { signal }),
    dataset: (id: string, signal?: Signal) => request<Dataset>(`${BASE}/datasets/${id}`, { signal }),
    /** A stored dataset's rows, its periods as chapkit writes them and in period order. */
    datasetData: (id: string, what: DataProjection = {}, signal?: Signal) =>
        request<Frame>(
            `${BASE}/datasets/${id}/$data${query({
                location: what.locations,
                column: what.columns,
                before: what.before,
                last: what.last,
                canonical: true,
            })}`,
            { signal },
        ),
    uploadDataset: (name: string, file: File, target?: string) => {
        const form = new FormData()
        form.append('name', name)
        if (target !== undefined && target.trim() !== '') form.append('target', target.trim())
        form.append('file', file)
        return request<Dataset>(`${BASE}/datasets`, { method: 'POST', body: form })
    },
    deleteDataset: (id: string) => request<void>(`${BASE}/datasets/${id}`, { method: 'DELETE' }),

    backtest: (body: BacktestRequest) => request<JobSubmitted>(`${BASE}/ml/$backtest`, json('POST', body)),
    forecast: (body: ForecastRequest) => request<JobSubmitted>(`${BASE}/ml/$forecast`, json('POST', body)),
}

export type { Evaluation, Forecast }
