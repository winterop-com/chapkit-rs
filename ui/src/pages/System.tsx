import { DashboardGrid, DashboardPanel } from '@winterop-com/sokkel-ui/dashboard/DashboardGrid'
import { StatPanel } from '@winterop-com/sokkel-ui/dashboard/StatPanel'
import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { DescriptionList } from '@winterop-com/sokkel-ui/DescriptionList'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { StatusChip } from '@winterop-com/sokkel-ui/StatusChip'
import { useMemo } from 'react'

import { api } from '../api/client'
import { useModel } from '../App'
import { when } from '../lib/format'
import { useResource } from '../lib/resource'

/** One route's requests, from the service's Prometheus metrics. */
interface RouteLoad {
    key: string
    method: string
    route: string
    status: string
    requests: number
    seconds: number
}

/** The labels and value of each sample of a Prometheus text exposition. */
function samples(text: string) {
    const out: { name: string; labels: Record<string, string>; value: number }[] = []
    for (const line of text.split('\n')) {
        const match = /^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+(\S+)/.exec(line)
        if (match === null) continue
        const labels: Record<string, string> = {}
        for (const pair of (match[2] ?? '').matchAll(/(\w+)="((?:[^"\\]|\\.)*)"/g)) labels[pair[1] ?? ''] = pair[2] ?? ''
        out.push({ name: match[1] ?? '', labels, value: Number(match[3]) })
    }
    return out
}

/** How long the service has run, in its largest unit. */
function uptime(seconds: number): string {
    if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} min`
    if (seconds < 86_400) return `${(seconds / 3600).toFixed(1)} h`
    return `${(seconds / 86_400).toFixed(1)} days`
}

async function readMetrics(): Promise<string> {
    const response = await fetch('/metrics')
    if (!response.ok) throw new Error(`/metrics answered ${response.status}`)
    return response.text()
}

const LOAD: DataColumn<RouteLoad>[] = [
    { id: 'route', header: 'Route', width: 'minmax(16rem,2fr)', cell: (row) => <span className="font-mono text-xs">{row.route}</span>, sortValue: (row) => row.route },
    { id: 'method', header: 'Method', width: '6rem', cell: (row) => <span className="font-mono text-xs">{row.method}</span>, sortValue: (row) => row.method },
    { id: 'status', header: 'Status', width: '5rem', cell: (row) => <span className="font-mono text-xs">{row.status}</span>, sortValue: (row) => row.status },
    { id: 'requests', header: 'Requests', width: '6rem', align: 'end', cell: (row) => <span className="tabular-nums">{row.requests}</span>, sortValue: (row) => row.requests },
    {
        id: 'mean',
        header: 'Mean time',
        width: '7rem',
        align: 'end',
        cell: (row) => <span className="tabular-nums">{row.requests === 0 ? '-' : `${((row.seconds / row.requests) * 1000).toFixed(2)} ms`}</span>,
        sortValue: (row) => (row.requests === 0 ? 0 : row.seconds / row.requests),
    },
]

/** System: the host, the service's health checks, the model's facts, and what it has served. */
export function System() {
    const model = useModel()
    const system = useResource(api.system, [])
    const health = useResource(api.health, [], 10_000)
    const metrics = useResource(readMetrics, [], 10_000)
    const parsed = useMemo(() => samples(metrics.data ?? ''), [metrics.data])
    const load = useMemo(() => {
        const rows = new Map<string, RouteLoad>()
        for (const sample of parsed) {
            if (sample.name !== 'http_server_requests_total' && sample.name !== 'http_server_request_duration_seconds_sum') continue
            const { method = '', route = '', status = '' } = sample.labels
            const key = `${method} ${route} ${status}`
            const row = rows.get(key) ?? { key, method, route, status, requests: 0, seconds: 0 }
            if (sample.name === 'http_server_requests_total') row.requests += sample.value
            else row.seconds += sample.value
            rows.set(key, row)
        }
        return [...rows.values()]
    }, [parsed])
    const total = (name: string) => parsed.filter((sample) => sample.name === name).reduce((sum, sample) => sum + sample.value, 0)
    const started = parsed.find((sample) => sample.name === 'process_start_time_seconds')?.value
    const checks = Object.entries(health.data?.checks ?? {})

    return (
        <>
            <PageHeader
                title="System"
                description="The host the service runs on, its health and what it has served."
                lead={health.data ? <StatusChip status={health.data.status} /> : undefined}
            />
            <DashboardGrid>
                <StatPanel title="Requests" value={metrics.data ? total('http_server_requests_total') : null} loading={metrics.loading} span="quarter" />
                <StatPanel title="Evaluations run" value={metrics.data ? total('ml_backtest_jobs_total') : null} loading={metrics.loading} span="quarter" />
                <StatPanel title="Jobs finished" value={metrics.data ? total('jobs_finished_total') : null} loading={metrics.loading} span="quarter" />
                <StatPanel
                    title="Up for"
                    value={started === undefined ? null : Date.now() / 1000 - started}
                    format={uptime}
                    loading={metrics.loading}
                    span="quarter"
                />
                <DashboardPanel title="Host" span="half" loading={system.loading} error={system.error?.message}>
                    <DescriptionList
                        termWidth="9rem"
                        items={[
                            { term: 'Host name', value: system.data?.hostname, code: true },
                            { term: 'Platform', value: system.data?.platform, code: true },
                            { term: 'Time', value: when(system.data?.current_time) },
                            { term: 'Time zone', value: system.data?.timezone },
                            { term: 'chapkit-rs', value: system.data?.chapkit_rs_version, code: true },
                            { term: 'Chap API', value: model?.chapkit_version, code: true },
                        ]}
                    />
                </DashboardPanel>
                <DashboardPanel title="Health checks" span="half" loading={health.loading} error={health.error?.message}>
                    <DescriptionList
                        termWidth="9rem"
                        items={[
                            { term: 'Service', value: health.data ? <StatusChip status={health.data.status} /> : undefined },
                            ...checks.map(([name, check]) => ({ term: name.charAt(0).toUpperCase() + name.slice(1), value: <StatusChip status={check.state} /> })),
                            { term: 'Extensions', value: model?.extensions?.join(', '), code: true },
                            { term: 'Periods', value: model ? `${model.min_prediction_periods ?? 0} to ${model.max_prediction_periods ?? '-'}` : undefined },
                        ]}
                    />
                </DashboardPanel>
                <DashboardPanel
                    title="Requests by route"
                    subtitle="since the service started"
                    span="full"
                    flush
                    loading={metrics.loading}
                    error={metrics.error?.message}
                    empty={metrics.data !== undefined && load.length === 0}
                >
                    <DataTable
                        variant="list"
                        framed={false}
                        columns={LOAD}
                        rows={load}
                        rowKey={(row) => row.key}
                        defaultSort={{ column: 'requests', direction: 'desc' }}
                        minWidth="auto"
                        stackBelow={480}
                    />
                </DashboardPanel>
            </DashboardGrid>
        </>
    )
}
