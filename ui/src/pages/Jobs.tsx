import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { Problem } from '@winterop-com/sokkel-ui/Problem'
import { Segmented } from '@winterop-com/sokkel-ui/Segmented'
import { StatusChip } from '@winterop-com/sokkel-ui/StatusChip'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { RefreshCw } from 'lucide-react'
import { useState } from 'react'

import { api } from '../api/client'
import type { Job } from '../api/types'
import { dateOf, took, when } from '../lib/format'
import { useResource } from '../lib/resource'

const LIVE = new Set(['pending', 'running', 'canceling'])

const COLUMNS: DataColumn<Job>[] = [
    { id: 'id', header: 'Job', width: 'minmax(14rem,1.5fr)', cell: (row) => <span className="font-mono text-xs">{row.id}</span>, sortValue: (row) => row.id },
    { id: 'status', header: 'Status', width: '8rem', cell: (row) => <StatusChip status={row.status} />, sortValue: (row) => row.status },
    { id: 'submitted', header: 'Submitted', width: '11rem', cell: (row) => when(row.submitted_at), sortValue: (row) => dateOf(row.submitted_at).getTime() },
    {
        id: 'took',
        header: 'Took',
        width: '6rem',
        align: 'end',
        cell: (row) => <span className="tabular-nums">{took(row.started_at, row.finished_at)}</span>,
    },
    {
        id: 'error',
        header: 'Error',
        width: 'minmax(10rem,2fr)',
        cell: (row) => <span className="truncate text-muted-foreground">{row.error ?? ''}</span>,
    },
]

type Shown = 'all' | 'live' | 'failed'

/** Jobs: what the service ran and is running, refreshed every second or two while any runs. */
export function Jobs() {
    const [every, setEvery] = useState<number | undefined>(5_000)
    const jobs = useResource(api.jobs, [], every)
    const rows = jobs.data ?? []
    const live = rows.some((job) => LIVE.has(job.status))
    const wanted = live ? 1_500 : 5_000
    if (every !== wanted) setEvery(wanted)
    const [shown, setShown] = useState<Shown>('all')
    const listed = rows.filter((job) => (shown === 'all' ? true : shown === 'live' ? LIVE.has(job.status) : job.status === 'failed'))
    return (
        <>
            <PageHeader
                title="Jobs"
                count={jobs.data?.length}
                description={live ? 'Refreshing while jobs run.' : 'Evaluations, forecasts and trainings the service has run.'}
                actions={
                    <Button size="sm" variant="outline" onClick={jobs.reload}>
                        <RefreshCw aria-hidden />
                        Refresh
                    </Button>
                }
            />
            {jobs.error ? (
                <Problem title="Could not read the jobs" technical={jobs.error.message} />
            ) : (
                <DataTable
                    columns={COLUMNS}
                    rows={listed}
                    rowKey={(row) => row.id}
                    loading={jobs.loading}
                    defaultSort={{ column: 'submitted', direction: 'desc' }}
                    rowTone={(row) => (row.status === 'failed' ? 'critical' : undefined)}
                    toolbar={
                        <Segmented
                            label="Show"
                            size="sm"
                            value={shown}
                            onValueChange={setShown}
                            options={[
                                { value: 'all', label: 'All', count: rows.length },
                                { value: 'live', label: 'Running', count: rows.filter((job) => LIVE.has(job.status)).length, dot: 'var(--status-running)' },
                                { value: 'failed', label: 'Failed', count: rows.filter((job) => job.status === 'failed').length, dot: 'var(--status-failed)' },
                            ]}
                        />
                    }
                    paging="client"
                    defaultPageSize={50}
                    minWidth="48rem"
                    detail={(row) => (row.error_traceback ? <pre className="overflow-x-auto font-mono text-xs whitespace-pre-wrap">{row.error_traceback}</pre> : null)}
                    hasDetail={(row) => Boolean(row.error_traceback)}
                    empty="No job has run yet."
                />
            )}
        </>
    )
}
