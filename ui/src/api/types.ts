/** The wire types of the model service's API, as `openapi.json` describes them. */

/** A chapkit DataFrame: columns and rows of values. */
export interface Frame {
    columns: string[]
    data: (string | number | null)[][]
}

export interface ModelMetadata {
    author?: string | null
    author_assessed_status?: string | null
    author_note?: string | null
    contact_email?: string | null
    documentation_url?: string | null
    repository_url?: string | null
    organization?: string | null
    citation_info?: string | null
}

/** `GET /api/v1/info`. */
export interface Info {
    id: string
    display_name: string
    version: string
    description?: string | null
    model_metadata?: ModelMetadata | null
    period_type: string
    target?: string | null
    required_covariates: string[]
    chapkit_rs_version?: string
    chapkit_version?: string
    extensions?: string[]
    min_prediction_periods?: number
    max_prediction_periods?: number
    requires_geo?: boolean
    /** The largest request body the service takes, in bytes. chapkit-rs. */
    max_body_bytes?: number
}

export interface SystemInfo {
    current_time: string
    timezone: string
    chapkit_rs_version: string
    platform: string
    hostname: string
}

export type HealthState = 'healthy' | 'degraded' | 'unhealthy'

export interface Health {
    status: HealthState
    checks?: Record<string, { state: HealthState; message?: string | null }> | null
}

export interface Config {
    id: string
    name: string
    data: Record<string, unknown>
    tags: string[]
    created_at: string
    updated_at: string
}

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'canceling' | 'canceled'

export interface Job {
    id: string
    status: JobStatus
    submitted_at: string
    started_at?: string | null
    finished_at?: string | null
    error?: string | null
    error_traceback?: string | null
}

export interface JobSubmitted {
    job_id: string
    artifact_id: string
    message: string
}

export interface Artifact<C = unknown, S = unknown> {
    id: string
    parent_id?: string | null
    level: number
    tags: string[]
    created_at: string
    updated_at: string
    data: {
        type?: string
        content?: C
        content_type?: string | null
        content_size?: number | null
        metadata?: Record<string, unknown> | null
        /** A backtest's or forecast's summary, in a listing in place of its content. */
        summary?: S
    }
}

export interface ArtifactNode extends Artifact {
    children?: ArtifactNode[] | null
    hierarchy?: string | null
    level_label?: string | null
}

/** A stored dataset, `GET /api/v1/datasets/{id}`. */
export interface Dataset {
    id: string
    name: string
    created_at: string
    updated_at: string
    target?: string | null
    rows: number
    locations: number
    location_ids: string[]
    period_type: string
    first_period?: string | null
    last_period?: string | null
    periods: number
    columns: string[]
    covariates: string[]
    /** How a generated dataset was made; absent for an upload. */
    generator?: GeneratorSettings
}

/** The sample-data generator's settings, as `POST /api/v1/datasets/$generate` takes them. */
export interface GenerateRequest {
    name: string
    num_locations: number
    num_periods: number
    num_features: number
    period_type: 'monthly' | 'weekly'
    config_id?: string
    /** Absent: the service draws one and records it. */
    seed?: number
}

/** The settings a generated dataset was made with, as resolved, the seed always given. */
export type GeneratorSettings = Omit<GenerateRequest, 'name' | 'seed'> & { seed: number }

/** The scores chap-core computes, by metric id. */
export type Metrics = Partial<Record<MetricId, number | null>>

export type MetricId =
    | 'crps'
    | 'crps_log1p'
    | 'crps_norm'
    | 'mae'
    | 'rmse'
    | 'mape'
    | 'coverage_10_90'
    | 'coverage_25_75'
    | 'winkler_score_10_90'
    | 'winkler_score_25_75'
    | 'winkler_score_10_90_log1p'
    | 'winkler_score_25_75_log1p'
    | 'ratio_above_truth'

export interface BacktestOptions {
    periods?: number
    splits?: number
    stride?: number
    retrain?: number
    weather?: string
    drop_untrained?: boolean
    fill_gaps?: boolean
    pad?: boolean
}

/** One forecast of a backtest against what was observed. */
export interface BacktestForecast {
    split: number
    location: string
    time_period: string
    horizon: number
    observed: number | null
    median: number | null
    q10: number | null
    q25: number | null
    q75: number | null
    q90: number | null
}

/** The content of an `ml_backtest` artifact. */
export interface Evaluation {
    model: string
    options: BacktestOptions
    config?: Record<string, unknown>
    data: {
        first?: string
        last?: string
        locations?: number
        periods?: number
        rows?: number
        source?: string
        unit?: string
        dropped?: string[]
    }
    dataset_id?: string | null
    rows: number
    elapsed_ms: number
    aggregate: Metrics
    by_horizon: { horizon: number; metrics: Metrics }[]
    by_location: { location: string; metrics: Metrics }[]
    split_last?: string[]
    train_last?: string
    forecasts?: BacktestForecast[]
}

export interface ForecastRow {
    location: string
    time_period: string
    median: number | null
    q10: number | null
    q25: number | null
    q75: number | null
    q90: number | null
}

/** The content of an `ml_forecast` artifact. */
export interface Forecast {
    dataset_id: string
    config?: Record<string, unknown>
    periods: number
    weather?: string | null
    forecasts: ForecastRow[]
    frame?: Frame
}

/** What a listing says of an `ml_backtest` artifact. */
export type EvaluationSummary = Pick<Evaluation, 'dataset_id' | 'data' | 'options' | 'aggregate'>

/** What a listing says of an `ml_forecast` artifact. */
export interface ForecastSummary {
    dataset_id?: string | null
    periods?: number
    weather?: string | null
    first_period?: string | null
    locations?: number
}

/** One page of a listing. */
export interface Page<T> {
    items: T[]
    total: number
    page: number
    size: number
    pages: number
}

/** A problem+json error body. */
export interface ProblemBody {
    type?: string
    title?: string
    status?: number
    detail?: string | null
}
