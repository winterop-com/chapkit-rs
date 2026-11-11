import type {
    Artifact,
    ArtifactNode,
    BacktestOptions,
    Config,
    Dataset,
    Evaluation,
    Forecast,
    Frame,
    Health,
    Info,
    Job,
    JobSubmitted,
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response
    try {
        response = await fetch(path, { ...init, headers: { accept: 'application/json', ...init?.headers } })
    } catch (error) {
        throw new ApiError('unreachable', 'The service does not answer', String(error))
    }
    if (!response.ok) {
        let body: ProblemBody = {}
        try {
            body = (await response.json()) as ProblemBody
        } catch {
            // Not problem+json: the status says it.
        }
        throw new ApiError(response.status, body.title ?? response.statusText, body.detail ?? undefined)
    }
    if (response.status === 204) return undefined as T
    return (await response.json()) as T
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

export const api = {
    info: () => request<Info>(`${BASE}/info`),
    system: () => request<SystemInfo>(`${BASE}/system`),
    health: () => request<Health>('/health'),
    openapi: () => request<{ paths: Record<string, Record<string, { summary?: string; tags?: string[] }>> }>('/openapi.json'),

    configs: async () => items(await request<Config[] | { items: Config[] }>(`${BASE}/configs`)),
    configSchema: () => request<Record<string, unknown>>(`${BASE}/configs/$schema`),
    createConfig: (name: string, data: unknown) => request<Config>(`${BASE}/configs`, json('POST', { name, data })),
    deleteConfig: (id: string) => request<void>(`${BASE}/configs/${id}`, { method: 'DELETE' }),

    jobs: () => request<Job[]>(`${BASE}/jobs`),
    job: (id: string) => request<Job>(`${BASE}/jobs/${id}`),

    /** The artifacts, of one type when given; a list carries no content, which `artifact` reads. */
    artifacts: async (type?: string) =>
        items(await request<Artifact[] | { items: Artifact[] }>(`${BASE}/artifacts${type === undefined ? '' : `?type=${type}`}`)),
    /** The newest artifacts of a type with their content, at most `limit` of them. */
    recent: async <C = unknown>(type: string, limit = 25) => {
        const listed = items(await request<Artifact[] | { items: Artifact[] }>(`${BASE}/artifacts?type=${type}`))
        const newest = [...listed].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, limit)
        return Promise.all(newest.map((each) => request<Artifact<C>>(`${BASE}/artifacts/${each.id}`)))
    },
    artifact: <C = unknown>(id: string) => request<Artifact<C>>(`${BASE}/artifacts/${id}`),
    artifactTree: (id: string) => request<ArtifactNode>(`${BASE}/artifacts/${id}/$tree`),
    artifactDownload: (id: string) => `${BASE}/artifacts/${id}/$download`,
    deleteArtifact: (id: string) => request<void>(`${BASE}/artifacts/${id}`, { method: 'DELETE' }),

    datasets: async () => items(await request<Dataset[] | { items: Dataset[] }>(`${BASE}/datasets`)),
    dataset: (id: string) => request<Dataset>(`${BASE}/datasets/${id}`),
    datasetData: (id: string, locations?: readonly string[]) => {
        const query = (locations ?? []).map((location) => `location=${encodeURIComponent(location)}`).join('&')
        return request<Frame>(`${BASE}/datasets/${id}/$data${query === '' ? '' : `?${query}`}`)
    },
    uploadDataset: (name: string, file: File) => {
        const form = new FormData()
        form.append('name', name)
        form.append('file', file)
        return request<Dataset>(`${BASE}/datasets`, { method: 'POST', body: form })
    },
    deleteDataset: (id: string) => request<void>(`${BASE}/datasets/${id}`, { method: 'DELETE' }),

    backtest: (body: BacktestRequest) => request<JobSubmitted>(`${BASE}/ml/$backtest`, json('POST', body)),
    forecast: (body: ForecastRequest) => request<JobSubmitted>(`${BASE}/ml/$forecast`, json('POST', body)),
}

export type { Evaluation, Forecast }
