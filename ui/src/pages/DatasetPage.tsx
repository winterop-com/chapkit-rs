import { Chart } from '@winterop-com/sokkel-ui-charts/Chart'
import { ChartFrame } from '@winterop-com/sokkel-ui-charts/ChartFrame'
import { chartLegend } from '@winterop-com/sokkel-ui-charts/chart-legend'
import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { DescriptionList } from '@winterop-com/sokkel-ui/DescriptionList'
import { ConfirmDialog } from '@winterop-com/sokkel-ui/DialogShell'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { PageState } from '@winterop-com/sokkel-ui/PageState'
import { Section } from '@winterop-com/sokkel-ui/Section'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { toast } from '@winterop-com/sokkel-ui/primitives/toast'
import { useEffect, useMemo, useState } from 'react'

import { api } from '../api/client'
import type { Frame } from '../api/types'
import { count, periodWord, when } from '../lib/format'
import { useResource } from '../lib/resource'
import { go } from '../lib/router'
import { LocationPicker } from '../parts'

type Row = Record<string, string | number | null> & { _key: string }

/** The rows of several frames as records, in the order given. */
function records(frames: readonly Frame[]): Row[] {
    return frames.flatMap((frame) =>
        frame.data.map((cells, index) => {
            const row: Row = { _key: '' }
            frame.columns.forEach((column, at) => {
                row[column] = cells[at] ?? null
            })
            row._key = `${String(row.location)}:${String(row.time_period)}:${index}`
            return row
        }),
    )
}

/**
 * One dataset: its facts, its target over time for the locations chosen, and its rows. Mounted
 * afresh for each dataset (the app keys pages by route), so a choice of locations never carries over.
 */
export function DatasetPage({ id }: { id: string }) {
    const dataset = useResource((signal) => api.dataset(id, signal), [id])
    const meta = dataset.data
    const [chosen, setChosen] = useState<string[]>([])
    const [removing, setRemoving] = useState(false)
    useEffect(() => {
        if (meta !== undefined && chosen.length === 0) setChosen(meta.location_ids.slice(0, 3))
        // Only the first locations, once the dataset has come.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [meta])
    // The periods come as the service writes them, in period order: drawn as they come.
    const frames = useResource(
        chosen.length === 0 ? null : (signal) => api.datasetData(id, { locations: chosen }, signal),
        [id, chosen.join('\u0000')],
    )
    const rows = useMemo(() => records(frames.data === undefined ? [] : [frames.data]), [frames.data])
    const target = meta?.target ?? 'disease_cases'
    const spec = useMemo(
        () => ({
            rows: rows.map((row) => ({ period: String(row.time_period), location: String(row.location), value: row[target] ?? null })),
            x: 'period',
            y: 'value',
            series: 'location',
            order: [...new Set(rows.map((row) => String(row.time_period)))],
            seriesOrder: chosen,
        }),
        [rows, target, chosen],
    )
    const legend = useMemo(() => chartLegend('line', spec), [spec])
    const columns: DataColumn<Row>[] = useMemo(
        () =>
            (meta?.columns ?? []).map((column, index) => ({
                id: column,
                header: column,
                width: index === 0 ? 'minmax(10rem,1.5fr)' : 'minmax(7rem,1fr)',
                align: column === 'time_period' || column === 'location' || column === 'parent' ? 'start' : 'end',
                cell: (row: Row) => {
                    const value = row[column]
                    if (value === null || value === undefined) return <span className="text-muted-foreground">-</span>
                    return typeof value === 'number' ? <span className="tabular-nums">{Number.isInteger(value) ? value : value.toFixed(2)}</span> : <span className="font-mono text-xs">{value}</span>
                },
                sortValue: (row: Row) => row[column] ?? '',
            })),
        [meta],
    )

    const remove = () =>
        api.deleteDataset(id).then(
            () => {
                toast.add({ title: `${meta?.name ?? 'Dataset'} deleted` })
                go('/datasets')
            },
            (failure: Error) => toast.add({ type: 'error', title: 'Could not delete the dataset', description: failure.message }),
        )

    return (
        <PageState loading={dataset.loading} error={dataset.error?.detail ?? dataset.error?.message} status={dataset.error?.status} onRetry={dataset.reload}>
            {meta && (
                <div className="flex flex-col gap-6">
                    <PageHeader
                        title={meta.name}
                        code={meta.id}
                        onBack={() => go('/datasets')}
                        meta={`Uploaded ${when(meta.created_at)}`}
                        actions={
                            <>
                                <Button variant="outline" size="sm" onClick={() => setRemoving(true)}>
                                    Delete
                                </Button>
                                <Button variant="outline" size="sm" onClick={() => go(`/forecast?dataset=${meta.id}`)}>
                                    Forecast
                                </Button>
                                <Button size="sm" onClick={() => go(`/evaluate?dataset=${meta.id}`)}>
                                    Evaluate
                                </Button>
                            </>
                        }
                    />
                    <Section title="Facts">
                        <DescriptionList
                            columns={2}
                            items={[
                                { term: 'Period type', value: periodWord(meta.period_type) },
                                { term: 'Range', value: `${meta.first_period ?? '-'} to ${meta.last_period ?? '-'}`, code: true },
                                { term: 'Periods', value: count(meta.periods), numeric: true },
                                { term: 'Locations', value: count(meta.locations), numeric: true },
                                { term: 'Rows', value: count(meta.rows), numeric: true },
                                { term: 'Target', value: meta.target, code: true },
                                { term: 'Covariates', value: meta.covariates.length === 0 ? 'None' : meta.covariates.join(', '), wide: true },
                                { term: 'Columns', value: meta.columns.join(', '), code: true, wide: true },
                            ]}
                        />
                    </Section>
                    <ChartFrame
                        bodyClassName="h-72"
                        title={`${target} over time`}
                        subtitle={`${chosen.length} of ${meta.locations} locations`}
                        legend={legend}
                        file={`${meta.name}-${target}`}
                        loading={frames.loading}
                        error={frames.error?.message}
                        technical={frames.error?.detail}
                        onRetry={frames.reload}
                        empty={chosen.length === 0}
                        actions={<LocationPicker locations={meta.location_ids} value={chosen} onChange={setChosen} />}
                    >
                        {(state) => <Chart kind="line" label={`${target} over time by location`} className="h-full" {...spec} {...state} />}
                    </ChartFrame>
                    <Section title="Rows" meta={`${count(rows.length)} rows of the locations shown`} flush>
                        <DataTable
                            columns={columns}
                            rows={rows}
                            rowKey={(row) => row._key}
                            loading={frames.loading}
                            framed={false}
                            paging="client"
                            defaultPageSize={25}
                            pageSizes={[25, 50, 100]}
                            bar={false}
                            minWidth="40rem"
                            empty={frames.error ? `Could not read the rows: ${frames.error.message}` : 'Choose a location to see its rows.'}
                        />
                    </Section>
                    <ConfirmDialog
                        open={removing}
                        onOpenChange={setRemoving}
                        question={`Delete ${meta.name}?`}
                        detail="Evaluations and forecasts made from it are kept."
                        confirm={{ label: 'Delete', destructive: true, onConfirm: () => void remove() }}
                    />
                </div>
            )}
        </PageState>
    )
}
