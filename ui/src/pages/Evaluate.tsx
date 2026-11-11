import { ChartPanel } from '@winterop-com/sokkel-ui-charts/dashboard/ChartPanel'
import { DashboardGrid, DashboardPanel } from '@winterop-com/sokkel-ui/dashboard/DashboardGrid'
import { ScoreTable } from '@winterop-com/sokkel-ui/dashboard/ScoreTable'
import { StatPanel } from '@winterop-com/sokkel-ui/dashboard/StatPanel'
import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { DescriptionList } from '@winterop-com/sokkel-ui/DescriptionList'
import { Stack } from '@winterop-com/sokkel-ui/Layout'
import { NumberField } from '@winterop-com/sokkel-ui/NumberField'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { PageState } from '@winterop-com/sokkel-ui/PageState'
import { Problem } from '@winterop-com/sokkel-ui/Problem'
import { Section } from '@winterop-com/sokkel-ui/Section'
import { Segmented } from '@winterop-com/sokkel-ui/Segmented'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { Download, Play } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { api } from '../api/client'
import type { Artifact, Evaluation, MetricId } from '../api/types'
import { canonicalPeriod, count, dateOf, METRICS, metricName, score, when } from '../lib/format'
import { useResource } from '../lib/resource'
import { go, hashParam } from '../lib/router'
import { configOptions, datasetOptions, JobProgress, LocationFan, LocationPicker, SelectField, WEATHER, type FanPoint } from '../parts'

const TILES: { id: MetricId; target?: number }[] = [{ id: 'crps' }, { id: 'mae' }, { id: 'rmse' }, { id: 'coverage_10_90', target: 80 }]
const CHART_METRICS = ['crps', 'mae', 'rmse', 'coverage_10_90'] as const
const NONE: FanPoint[] = []
const HORIZON_METRICS = METRICS.filter((metric) => (CHART_METRICS as readonly string[]).includes(metric.id))

/** The form that starts a backtest, and the job it started until it ends. */
function NewEvaluation({ onDone }: { onDone: (artifactId: string) => void }) {
    const datasets = useResource(api.datasets, [])
    const configs = useResource(api.configs, [])
    const [dataset, setDataset] = useState(hashParam('dataset') ?? '')
    const [config, setConfig] = useState('')
    const [periods, setPeriods] = useState<number | null>(3)
    const [splits, setSplits] = useState<number | null>(7)
    const [stride, setStride] = useState<number | null>(1)
    const [weather, setWeather] = useState('climatology')
    const [running, setRunning] = useState<{ job: string; artifact: string }>()
    const [error, setError] = useState<{ title: string; detail?: string }>()
    useEffect(() => {
        const first = datasets.data?.[0]
        if (dataset === '' && first !== undefined) setDataset(first.id)
    }, [datasets.data, dataset])

    const start = () => {
        setError(undefined)
        api.backtest({
            dataset_id: dataset,
            ...(config === '' ? {} : { config_id: config }),
            ...(periods === null ? {} : { periods }),
            ...(splits === null ? {} : { splits }),
            ...(stride === null ? {} : { stride }),
            weather,
        }).then(
            (submitted) => setRunning({ job: submitted.job_id, artifact: submitted.artifact_id }),
            (failure: { message: string; detail?: string }) => setError({ title: failure.message, detail: failure.detail }),
        )
    }
    const noData = datasets.data !== undefined && datasets.data.length === 0
    return (
        <Section title="New evaluation" description="Backtest the model on a dataset: train on the past, forecast the next periods, and score against what happened, split after split.">
            <Stack gap="lg">
                {noData ? (
                    <Problem
                        tone="warning"
                        title="There is no dataset to evaluate on"
                        detail="Upload one first."
                        actions={
                            <Button size="sm" variant="outline" onClick={() => go('/datasets')}>
                                Go to datasets
                            </Button>
                        }
                    />
                ) : (
                    <div className="grid gap-4 sm:grid-cols-2">
                        <SelectField id="eval-dataset" label="Dataset" value={dataset} options={datasetOptions(datasets.data)} onChange={setDataset} />
                        <SelectField id="eval-config" label="Config" value={config} placeholder="Model defaults" options={configOptions(configs.data)} onChange={setConfig} />
                        <NumberField id="eval-periods" label="Periods" hint="per forecast" integer min={1} value={periods} onValueChange={setPeriods} />
                        <NumberField id="eval-splits" label="Splits" integer min={1} value={splits} onValueChange={setSplits} />
                        <NumberField id="eval-stride" label="Stride" hint="periods between splits" integer min={1} value={stride} onValueChange={setStride} />
                        <SelectField id="eval-weather" label="Future weather" value={weather} options={WEATHER} onChange={setWeather} />
                    </div>
                )}
                <div className="flex items-center gap-3">
                    <Button onClick={start} disabled={dataset === '' || (running !== undefined && error === undefined)}>
                        <Play aria-hidden />
                        Run evaluation
                    </Button>
                </div>
                {error && <Problem title={`The evaluation was refused: ${error.title}`} detail={error.detail} />}
                {running && <JobProgress key={running.job} jobId={running.job} what="evaluation" onDone={() => onDone(running.artifact)} />}
            </Stack>
        </Section>
    )
}

/** The columns of past runs, naming each one's dataset. */
const pastColumns = (names: ReadonlyMap<string, string>): DataColumn<Artifact>[] => [
    { id: 'created', header: 'Created', width: '11rem', cell: (row) => when(row.created_at), sortValue: (row) => dateOf(row.created_at).getTime() },
    {
        id: 'dataset',
        header: 'Dataset',
        width: 'minmax(10rem,1.5fr)',
        cell: (row) => {
            const id = (row.data.content as Evaluation | undefined)?.dataset_id
            return id ? (names.get(id) ?? <span className="font-mono text-xs">{id}</span>) : <span className="text-muted-foreground">request data</span>
        },
    },
    {
        id: 'data',
        header: 'Data',
        width: 'minmax(10rem,1.5fr)',
        cell: (row) => {
            const data = (row.data.content as Evaluation | undefined)?.data
            return <span className="font-mono text-xs">{data?.first ? `${canonicalPeriod(data.first)} to ${canonicalPeriod(data.last ?? '')}` : '-'}</span>
        },
    },
    {
        id: 'setup',
        header: 'Setup',
        width: 'minmax(10.5rem,1fr)',
        cell: (row) => {
            const options = (row.data.content as Evaluation | undefined)?.options
            return options ? `${options.splits ?? '-'} splits x ${options.periods ?? '-'} periods` : '-'
        },
    },
    ...(['crps', 'mae', 'coverage_10_90'] as const).map(
        (id): DataColumn<Artifact> => ({
            id,
            header: metricName(id),
            width: '8.5rem',
            align: 'end',
            cell: (row) => <span className="tabular-nums">{score((row.data.content as Evaluation | undefined)?.aggregate[id], id.startsWith('coverage'))}</span>,
            sortValue: (row) => (row.data.content as Evaluation | undefined)?.aggregate[id] ?? Number.POSITIVE_INFINITY,
        }),
    ),
]

/** One evaluation's scores: tiles, by horizon, by location, and the forecasts against what happened. */
function EvaluationView({ id }: { id: string }) {
    const artifact = useResource(() => api.artifact<Evaluation>(id), [id])
    const result = artifact.data?.data.content
    const dataset = useResource(result?.dataset_id ? () => api.dataset(result.dataset_id!) : null, [result?.dataset_id])
    const [metric, setMetric] = useState<(typeof CHART_METRICS)[number]>('crps')
    const forecasts = useMemo(() => result?.forecasts ?? [], [result])
    const splits = useMemo(() => [...new Set(forecasts.map((row) => row.split))].sort((a, b) => a - b), [forecasts])
    const locations = useMemo(() => [...new Set(forecasts.map((row) => row.location))], [forecasts])
    const [split, setSplit] = useState<string>('')
    const [chosen, setChosen] = useState<string[]>([])
    useEffect(() => {
        if (splits.length > 0) setSplit(String(splits[splits.length - 1]))
        // The locations it scored worst on first: where a reader looks.
        const worst = [...(result?.by_location ?? [])]
            .sort((a, b) => (b.metrics.crps ?? 0) - (a.metrics.crps ?? 0))
            .map((row) => row.location)
            .filter((location) => locations.includes(location))
        setChosen((worst.length > 0 ? worst : locations).slice(0, 4))
    }, [splits, locations, result])

    const fans = useMemo(() => {
        const out = new Map<string, FanPoint[]>()
        for (const row of forecasts) {
            if (String(row.split) !== split) continue
            out.set(row.location, [...(out.get(row.location) ?? []), { ...row, period: row.time_period }])
        }
        return out
    }, [forecasts, split])
    const horizonRows = useMemo(
        () => (result?.by_horizon ?? []).map((row) => ({ horizon: String(row.horizon), value: row.metrics[metric] ?? null })),
        [result, metric],
    )
    const share = metric.startsWith('coverage')
    const locationColumns: DataColumn<{ location: string; metrics: Evaluation['aggregate'] }>[] = [
        { id: 'location', header: 'Location', width: 'minmax(13rem,1.5fr)', cell: (row) => <span className="font-mono text-xs">{row.location}</span>, sortValue: (row) => row.location },
        ...METRICS.slice(0, 6).map((each) => ({
            id: each.id,
            header: each.name,
            width: 'minmax(8.5rem,1fr)',
            align: 'end' as const,
            cell: (row: { metrics: Evaluation['aggregate'] }) => <span className="tabular-nums">{score(row.metrics[each.id], each.share)}</span>,
            sortValue: (row: { metrics: Evaluation['aggregate'] }) => row.metrics[each.id] ?? Number.POSITIVE_INFINITY,
        })),
    ]

    return (
        <PageState loading={artifact.loading} error={artifact.error?.detail ?? artifact.error?.message} status={artifact.error?.status}>
            {result && artifact.data && (
                <Stack gap="section">
                    <PageHeader
                        title="Evaluation"
                        code={id}
                        onBack={() => go('/evaluate')}
                        meta={`${when(artifact.data.created_at)}, ${dataset.data?.name ?? result.data.source ?? 'request data'}, ${count(result.rows)} scored forecasts in ${result.elapsed_ms.toFixed(1)} ms`}
                        actions={
                            <Button variant="outline" size="sm" nativeButton={false} render={<a href={api.artifactDownload(id)} download />}>
                                <Download aria-hidden />
                                Download
                            </Button>
                        }
                    />
                    <DescriptionList
                        layout="row"
                        items={[
                            { term: 'Data', value: result.data.first ? `${canonicalPeriod(result.data.first)} to ${canonicalPeriod(result.data.last ?? '')}` : '-', code: true },
                            { term: 'Locations', value: result.data.locations, numeric: true },
                            { term: 'Splits', value: result.options.splits, numeric: true },
                            { term: 'Periods', value: result.options.periods, numeric: true },
                            { term: 'Stride', value: result.options.stride, numeric: true },
                            { term: 'Weather', value: result.options.weather },
                        ]}
                    />
                    <DashboardGrid>
                        {TILES.map((tile) => {
                            const about = METRICS.find((each) => each.id === tile.id)
                            const value = result.aggregate[tile.id]
                            return (
                                <StatPanel
                                    key={tile.id}
                                    title={metricName(tile.id)}
                                    info={about?.info}
                                    span="quarter"
                                    value={value === null || value === undefined ? null : about?.share ? value * 100 : value}
                                    unit={about?.share ? '%' : undefined}
                                    format={about?.share ? undefined : (number) => score(number)}
                                    decimals={about?.share ? 1 : undefined}
                                    target={tile.target}
                                    targetRange={tile.target === undefined ? undefined : [0, 100]}
                                />
                            )
                        })}
                        <ChartPanel
                            title="By horizon"
                            subtitle={`${metricName(metric)}, by periods ahead`}
                            span="half"
                            kind="bar"
                            rows={horizonRows}
                            x="horizon"
                            y="value"
                            fieldNames={{ value: metricName(metric), horizon: 'Periods ahead' }}
                            format={(value: number) => score(value, share)}
                            legend="none"
                            file={`evaluation-${id}-by-horizon`}
                            actions={
                                <Segmented
                                    label="Score"
                                    size="sm"
                                    value={metric}
                                    onValueChange={setMetric}
                                    options={CHART_METRICS.map((each) => ({ value: each, label: metricName(each) }))}
                                />
                            }
                        />
                        <DashboardPanel title="Scores by horizon" subtitle={`${result.by_horizon.length} horizons`} span="half" flush>
                            <ScoreTable
                                framed={false}
                                nameHeader="Horizon"
                                columns={HORIZON_METRICS.map((each) => ({
                                    key: each.id,
                                    label: each.name,
                                    format: (value: number) => score(value, each.share),
                                }))}
                                rows={result.by_horizon.map((row) => ({
                                    key: String(row.horizon),
                                    label: `${row.horizon} ahead`,
                                    values: row.metrics,
                                }))}
                            />
                        </DashboardPanel>
                    </DashboardGrid>
                    <Section title="By location" meta={`${result.by_location.length} locations`} flush>
                        <DataTable
                            columns={locationColumns}
                            rows={result.by_location}
                            rowKey={(row) => row.location}
                            defaultSort={{ column: 'crps', direction: 'desc' }}
                            framed={false}
                            bar={false}
                            minWidth="60rem"
                            paging={result.by_location.length > 50 ? 'client' : undefined}
                            defaultPageSize={25}
                        />
                    </Section>
                    <Section
                        title="Forecasts against what happened"
                        description="The median forecast with its 10-90 and 25-75 intervals, and the observations, for one split."
                        actions={
                            forecasts.length > 0 && (
                                <div className="flex flex-wrap items-center gap-2">
                                    <LocationPicker locations={locations} value={chosen} onChange={setChosen} />
                                    {splits.length > 1 && (
                                        <Segmented
                                            label="Split"
                                            size="sm"
                                            value={split}
                                            onValueChange={setSplit}
                                            options={splits.map((each) => ({ value: String(each), label: `Split ${each + 1}` }))}
                                        />
                                    )}
                                </div>
                            )
                        }
                    >
                        {forecasts.length === 0 ? (
                            <p className="text-sm text-muted-foreground">This evaluation kept no forecasts; evaluations run on a stored dataset keep them.</p>
                        ) : (
                            <div className="grid gap-4 lg:grid-cols-2">
                                {chosen.map((location) => (
                                    <LocationFan
                                        key={location}
                                        dataset={dataset.data}
                                        location={location}
                                        subtitle={`split ${Number(split) + 1}`}
                                        file={`evaluation-${location}`}
                                        forecast={fans.get(location) ?? NONE}
                                    />
                                ))}
                            </div>
                        )}
                    </Section>
                </Stack>
            )}
        </PageState>
    )
}

/** Evaluate: start a backtest, follow it, and read its scores; past evaluations below. */
export function Evaluate({ opened }: { opened: string | undefined }) {
    const history = useResource(() => api.recent('ml_backtest', 50), [opened])
    const datasets = useResource(api.datasets, [])
    const names = useMemo(() => new Map((datasets.data ?? []).map((each) => [each.id, each.name])), [datasets.data])
    if (opened !== undefined) return <EvaluationView id={opened} />
    return (
        <Stack gap="section">
            <PageHeader title="Evaluate" description="How well the model forecasts, scored as chap-core scores it." />
            <NewEvaluation onDone={(artifactId) => go(`/evaluate/${artifactId}`)} />
            <Section title="Past evaluations" meta={history.data ? `${history.data.length}` : undefined} flush>
                <DataTable
                    columns={pastColumns(names)}
                    rows={history.data ?? []}
                    rowKey={(row) => row.id}
                    loading={history.loading}
                    defaultSort={{ column: 'created', direction: 'desc' }}
                    onOpen={(row) => go(`/evaluate/${row.id}`)}
                    chevron
                    framed={false}
                    bar={false}
                    minWidth="48rem"
                    empty="No evaluation yet."
                />
            </Section>
        </Stack>
    )
}
