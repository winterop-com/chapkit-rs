import { Chart } from '@winterop-com/sokkel-ui-charts/Chart'
import { ChartFrame } from '@winterop-com/sokkel-ui-charts/ChartFrame'
import { chartLegend } from '@winterop-com/sokkel-ui-charts/chart-legend'
import { FilterChip } from '@winterop-com/sokkel-ui/FilterChip'
import { LoadingBar } from '@winterop-com/sokkel-ui/Loaders'
import { PageState, type PageStateTone } from '@winterop-com/sokkel-ui/PageState'
import { Problem } from '@winterop-com/sokkel-ui/Problem'
import { StatusChip } from '@winterop-com/sokkel-ui/StatusChip'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'

import { aborted, ApiError, api, asApiError } from './api/client'
import type { Config, Dataset, Job, JobSubmitted } from './api/types'
import { score, took } from './lib/format'
import { pollDelay, useResource, type Resource } from './lib/resource'

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

/**
 * A read's failure, said with what failed and a retry; the children while it has not failed. `warning`
 * draws a read of secondary data the page stands without, in the warning tone.
 */
export function ReadState({ what, resource, tone, children }: { what: string; resource: Resource<unknown>; tone?: PageStateTone; children?: ReactNode }) {
    const error = resource.error
    return (
        <PageState
            loading={false}
            error={error === undefined ? undefined : `Could not read ${what}: ${error.detail ?? error.message}`}
            status={error?.status}
            onRetry={resource.reload}
            tone={tone}
        >
            {children}
        </PageState>
    )
}

/** How a job the page started ended, when it did not complete. */
export type JobEnd = 'failed' | 'canceled' | 'gone'

/**
 * Follows a job until it ends: its state and how long it has run, a bar while it runs, and what
 * failed. Calls `onDone` once when it completes and `onEnd` when it fails, is canceled or is gone.
 * A read that fails is tried again, ever more slowly, and said meanwhile.
 */
export function JobProgress({ jobId, what, onDone, onEnd }: { jobId: string; what: string; onDone: () => void; onEnd: (end: JobEnd) => void }) {
    const [job, setJob] = useState<Job>()
    const [lost, setLost] = useState<ApiError>()
    const [gone, setGone] = useState(false)
    const [nudge, setNudge] = useState(0)
    const callbacks = useRef({ onDone, onEnd })
    callbacks.current = { onDone, onEnd }
    useEffect(() => {
        const controller = new AbortController()
        let timer: ReturnType<typeof setTimeout> | undefined
        let failures = 0
        const poll = async () => {
            try {
                const next = await api.job(jobId, controller.signal)
                failures = 0
                setLost(undefined)
                setJob(next)
                if (next.status === 'completed') callbacks.current.onDone()
                else if (next.status === 'failed' || next.status === 'canceled') callbacks.current.onEnd(next.status)
                else timer = setTimeout(() => void poll(), pollDelay(0))
            } catch (failure) {
                if (controller.signal.aborted || aborted(failure)) return
                const error = asApiError(failure)
                if (error.status === 404) {
                    setGone(true)
                    callbacks.current.onEnd('gone')
                    return
                }
                failures += 1
                setLost(error)
                timer = setTimeout(() => void poll(), pollDelay(failures))
            }
        }
        void poll()
        return () => {
            controller.abort()
            if (timer !== undefined) clearTimeout(timer)
        }
    }, [jobId, nudge])
    if (gone) return <Problem title={`The ${what} is gone`} detail={`The service no longer knows job ${jobId}; it may have restarted.`} />
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
            {lost !== undefined && (
                <Problem
                    tone="warning"
                    title={`Lost track of the ${what} for now; trying again`}
                    detail="The job keeps running on the service."
                    technical={lost.detail ?? lost.message}
                    actions={
                        <Button size="sm" variant="outline" onClick={() => setNudge((count) => count + 1)}>
                            Try now
                        </Button>
                    }
                />
            )}
        </div>
    )
}

/**
 * Starts one job at a time from a form: refuses a second submission while one is on its way or its
 * job runs, and frees the form again once the job has failed, been canceled or gone.
 */
export function useJobRun() {
    const [submitting, setSubmitting] = useState(false)
    const [running, setRunning] = useState<{ job: string; artifact: string }>()
    const [ended, setEnded] = useState(false)
    const [error, setError] = useState<{ title: string; detail?: string }>()
    const busy = useRef(false)
    const live = running !== undefined && !ended
    const start = (submit: () => Promise<JobSubmitted>) => {
        if (busy.current || live) return
        busy.current = true
        setSubmitting(true)
        setError(undefined)
        submit()
            .then(
                (submitted) => {
                    setRunning({ job: submitted.job_id, artifact: submitted.artifact_id })
                    setEnded(false)
                },
                (failure: unknown) => {
                    const refused = asApiError(failure)
                    setError({ title: refused.message, detail: refused.detail })
                },
            )
            .finally(() => {
                busy.current = false
                setSubmitting(false)
            })
    }
    return { start, submitting, running, live, error, busy: submitting || live, onEnd: () => setEnded(true) }
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

/** One period of a fan: the median and its intervals, what was observed, and where and how far ahead it was forecast. */
export type FanPoint = {
    period: string
    location?: string
    /** Periods ahead of the forecast's start, from 1; absent for the history before it. */
    ahead?: number
    observed?: number | null
    median?: number | null
    q10?: number | null
    q25?: number | null
    q75?: number | null
    q90?: number | null
}

/** A fan's fields as its legend, tooltips and data view name them. */
const FAN_NAMES = {
    period: 'Period',
    median: 'Median',
    observed: 'Observed',
    q10: '10th',
    q25: '25th',
    q75: '75th',
    q90: '90th',
    location: 'Location',
    ahead: 'Periods ahead',
}
/** What a fan's data view shows beside the drawn figures. */
const FAN_DATA_FIELDS = ['location', 'ahead']
const FAN_FORMAT = (value: number) => score(value)

const BANDS = [
    { low: 'q10', high: 'q90' },
    { low: 'q25', high: 'q75' },
]

/** What a fan draws from its points: periods in order, the fields named and figures written as scores, and where the forecast starts. */
export function fanSpec(points: readonly FanPoint[], start?: string) {
    const rows = [...points].sort((a, b) => a.period.localeCompare(b.period))
    const first = start ?? rows.find((row) => row.median !== undefined && row.median !== null)?.period
    return {
        rows,
        x: 'period',
        y: 'median',
        bands: BANDS,
        points: 'observed',
        history: 'observed',
        fieldNames: FAN_NAMES as Record<string, string>,
        dataFields: FAN_DATA_FIELDS,
        format: FAN_FORMAT,
        marks: first === undefined ? [] : [{ x: first, label: 'Forecast' }],
    }
}

/**
 * A forecast as a fan: the observed history as a line, then the median with its 10-90 and 25-75
 * intervals, and the observations as dots where there are any; `start` marks where the forecast
 * begins. Periods are as the service writes them, so in order as text.
 */
export function FanChart({
    title,
    subtitle,
    points,
    file,
    start,
    loading,
}: {
    title: string
    subtitle?: string
    points: readonly FanPoint[]
    file: string
    start?: string
    loading?: boolean
}) {
    const spec = useMemo(() => fanSpec(points, start), [points, start])
    const legend = useMemo(() => chartLegend('fan', spec), [spec])
    return (
        <ChartFrame bodyClassName="h-64" title={title} subtitle={subtitle} file={file} legend={legend} loading={loading} empty={points.length === 0}>
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

/** How many periods of history a fan shows before its forecast, at least. */
export const HISTORY_PERIODS = 12

/**
 * The observed history before a forecast, for some locations in one read: the dataset's target in
 * the last `HISTORY_PERIODS` periods before `first` (or four times the forecast's length), by
 * location. The service picks the periods and writes them, so nothing here reads a period.
 */
export function useHistory(
    dataset: Dataset | undefined,
    locations: readonly string[],
    first: string | undefined,
    length: number,
): Resource<ReadonlyMap<string, FanPoint[]>> {
    const target = dataset?.target ?? 'disease_cases'
    const last = Math.max(HISTORY_PERIODS, length * 4)
    return useResource(
        dataset === undefined || first === undefined || locations.length === 0
            ? null
            : async (signal) => {
                  const frame = await api.datasetData(dataset.id, { locations, columns: [target], before: first, last }, signal)
                  const period = frame.columns.indexOf('time_period')
                  const location = frame.columns.indexOf('location')
                  const value = frame.columns.indexOf(target)
                  const out = new Map<string, FanPoint[]>()
                  for (const cells of frame.data) {
                      const at = String(cells[location])
                      const observed = cells[value]
                      out.set(at, [...(out.get(at) ?? []), { period: String(cells[period]), location: at, observed: typeof observed === 'number' ? observed : null }])
                  }
                  return out
              },
        [dataset?.id, target, locations.join('\u0000'), first, last],
    )
}

/** The note over a page's fans when their history is not drawn: still reading, failed, or nowhere to read it. */
export function HistoryNote({ dataset, history }: { dataset: Resource<Dataset> | undefined; history: Resource<unknown> }) {
    if (dataset?.error?.status === 404)
        return <Problem tone="warning" title="The dataset this came from is gone" detail="The charts show the forecasts without the history before them." />
    if (dataset?.error !== undefined) return <ReadState what="the dataset" resource={dataset} tone="warning" />
    return <ReadState what="the observed history" resource={history} tone="warning" />
}
