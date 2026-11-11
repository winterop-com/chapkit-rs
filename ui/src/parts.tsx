import { Chart } from '@winterop-com/sokkel-ui-charts/Chart'
import { ChartFrame } from '@winterop-com/sokkel-ui-charts/ChartFrame'
import { chartLegend } from '@winterop-com/sokkel-ui-charts/chart-legend'
import { FilterChip } from '@winterop-com/sokkel-ui/FilterChip'
import { LoadingBar } from '@winterop-com/sokkel-ui/Loaders'
import { Problem } from '@winterop-com/sokkel-ui/Problem'
import { StatusChip } from '@winterop-com/sokkel-ui/StatusChip'
import { useEffect, useMemo, useState } from 'react'

import { api } from './api/client'
import type { Config, Dataset, Job } from './api/types'
import { canonicalPeriod, took } from './lib/format'
import { useResource } from './lib/resource'

/** A choice of one of a few values, as a labeled field (sokkel-ui 0.15.7). */
export { SelectField } from '@winterop-com/sokkel-ui/SelectField'

export const datasetOptions = (datasets: readonly Dataset[] | undefined) =>
    (datasets ?? []).map((dataset) => ({ value: dataset.id, label: `${dataset.name} (${dataset.locations} locations, ${dataset.periods} periods)` }))

export const configOptions = (configs: readonly Config[] | undefined) => [
    { value: '', label: 'Model defaults' },
    ...(configs ?? []).map((config) => ({ value: config.id, label: config.name })),
]

export const WEATHER = [
    { value: 'climatology', label: 'Climatology' },
    { value: 'damped_persistence', label: 'Damped persistence' },
    { value: 'mstl_arima', label: 'MSTL + ARIMA' },
    { value: 'observed', label: 'Observed (oracle)' },
]

const TERMINAL = new Set(['completed', 'failed', 'canceled'])

/**
 * Follows a job until it ends: its state and how long it has run, a bar while it runs, and what
 * failed. Calls `onDone` once when it completes.
 */
export function JobProgress({ jobId, onDone, what }: { jobId: string; onDone: () => void; what: string }) {
    const [job, setJob] = useState<Job>()
    const [error, setError] = useState<string>()
    useEffect(() => {
        let live = true
        let timer = 0
        const poll = async () => {
            try {
                const next = await api.job(jobId)
                if (!live) return
                setJob(next)
                if (next.status === 'completed') onDone()
                else if (!TERMINAL.has(next.status)) timer = window.setTimeout(() => void poll(), 400)
            } catch (failure) {
                if (live) setError(String(failure))
            }
        }
        void poll()
        return () => {
            live = false
            window.clearTimeout(timer)
        }
        // onDone is the caller's; the job is what is followed.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [jobId])
    if (error !== undefined) return <Problem title={`Could not follow the ${what}`} technical={error} />
    if (job?.status === 'failed' || job?.status === 'canceled')
        return <Problem title={`The ${what} ${job.status}`} detail={job.error ?? undefined} technical={job.error_traceback ?? undefined} />
    return (
        <div className="flex flex-col gap-2" aria-live="polite">
            <div className="flex items-center gap-3 text-sm">
                <StatusChip status={job?.status ?? 'pending'} />
                <span className="text-muted-foreground">
                    {what} <span className="font-mono">{jobId.slice(-8)}</span>
                    {job?.started_at ? `, ${took(job.started_at, job.finished_at ?? new Date().toISOString())}` : ''}
                </span>
            </div>
            <LoadingBar label={`Running the ${what}`} />
        </div>
    )
}

/** Picks some of a dataset's locations, searching once there are many. */
export function LocationPicker({
    locations,
    value,
    onChange,
}: {
    locations: readonly string[]
    value: readonly string[]
    onChange: (value: string[]) => void
}) {
    const options = useMemo(() => locations.map((location) => ({ value: location })), [locations])
    return <FilterChip facet="Locations" multiple search options={options} selected={value} onSelectedChange={onChange} />
}

/** One period of a fan: the median and its intervals, and what was observed. */
export type FanPoint = {
    period: string
    observed?: number | null
    median?: number | null
    q10?: number | null
    q25?: number | null
    q75?: number | null
    q90?: number | null
}

const BANDS = [
    { low: 'q10', high: 'q90' },
    { low: 'q25', high: 'q75' },
]

/**
 * A forecast as a fan: the observed history as a line, then the median with its 10-90 and 25-75
 * intervals, and the observations as dots where there are any; `start` marks where the forecast
 * begins.
 */
export function FanChart({
    title,
    subtitle,
    points,
    file,
    start,
}: {
    title: string
    subtitle?: string
    points: readonly FanPoint[]
    file: string
    start?: string
}) {
    const spec = useMemo(() => {
        const rows = [...points].sort((a, b) => a.period.localeCompare(b.period))
        const first = start ?? rows.find((row) => row.median !== undefined && row.median !== null)?.period
        return {
            rows,
            x: 'period',
            y: 'median',
            bands: BANDS,
            points: 'observed',
            history: 'observed',
            fieldNames: { median: 'Median', observed: 'Observed', period: 'Period' },
            marks: first === undefined ? [] : [{ x: first, label: 'Forecast' }],
        }
    }, [points, start])
    const legend = useMemo(() => chartLegend('fan', spec), [spec])
    return (
        <ChartFrame bodyClassName="h-64" title={title} subtitle={subtitle} file={file} legend={legend} empty={points.length === 0}>
            {(state) => (
                <Chart
                    kind="fan"
                    label={`${title}: forecast median with 10-90 and 25-75 intervals, and observations`}
                    className="h-full"
                    {...spec}
                    {...state}
                />
            )}
        </ChartFrame>
    )
}

/** How many periods of history a fan shows before its forecast. */
export const HISTORY_PERIODS = 12

/**
 * A location's fan: the dataset's observations for the periods before the forecast (the last
 * `HISTORY_PERIODS` of them, or four times the forecast's length), then the forecast's own rows.
 */
export function useFanPoints(dataset: Dataset | undefined, location: string, forecast: readonly FanPoint[]): FanPoint[] {
    const history = useResource(dataset === undefined ? null : () => api.datasetData(dataset.id, [location]), [dataset?.id, location])
    return useMemo(() => {
        const first = [...forecast].map((point) => point.period).sort()[0]
        const frame = history.data
        const target = dataset?.target ?? 'disease_cases'
        const past: FanPoint[] = []
        if (frame !== undefined && first !== undefined) {
            const period = frame.columns.indexOf('time_period')
            const value = frame.columns.indexOf(target)
            for (const cells of frame.data) {
                const at = canonicalPeriod(String(cells[period]))
                const observed = cells[value]
                if (at < first) past.push({ period: at, observed: typeof observed === 'number' ? observed : null })
            }
            past.sort((a, b) => a.period.localeCompare(b.period))
        }
        return [...past.slice(-Math.max(HISTORY_PERIODS, forecast.length * 4)), ...forecast]
    }, [history.data, forecast, dataset?.target])
}

/** One location's fan, its history read from the dataset. */
export function LocationFan({
    dataset,
    location,
    forecast,
    subtitle,
    file,
}: {
    dataset: Dataset | undefined
    location: string
    forecast: readonly FanPoint[]
    subtitle: string
    file: string
}) {
    const points = useFanPoints(dataset, location, forecast)
    return <FanChart title={location} subtitle={subtitle} points={points} file={file} />
}
