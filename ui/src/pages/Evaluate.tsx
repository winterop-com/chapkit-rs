import { ChartPanel } from '@winterop-com/sokkel-ui-charts/dashboard/ChartPanel'
import { DashboardGrid, DashboardPanel } from '@winterop-com/sokkel-ui/dashboard/DashboardGrid'
import { ScoreTable } from '@winterop-com/sokkel-ui/dashboard/ScoreTable'
import { StatPanel } from '@winterop-com/sokkel-ui/dashboard/StatPanel'
import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { DescriptionList } from '@winterop-com/sokkel-ui/DescriptionList'
import { useShortcuts } from '@winterop-com/sokkel-ui/hooks/use-shortcuts'
import { Stack } from '@winterop-com/sokkel-ui/Layout'
import { NumberField } from '@winterop-com/sokkel-ui/NumberField'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { PageState } from '@winterop-com/sokkel-ui/PageState'
import { Problem } from '@winterop-com/sokkel-ui/Problem'
import { Section } from '@winterop-com/sokkel-ui/Section'
import { Segmented } from '@winterop-com/sokkel-ui/Segmented'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { toast } from '@winterop-com/sokkel-ui/primitives/toast'
import { Download, Play } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { api } from '../api/client'
import type { Artifact, BacktestForecast, Evaluation, EvaluationSummary, MetricId } from '../api/types'
import { count, METRICS, metricName, score, when } from '../lib/format'
import { useResource } from '../lib/resource'
import { go, hashParam } from '../lib/router'
import {
    configOptions,
    datasetOptions,
    FanChart,
    HistoryNote,
    JobProgress,
    LocationPicker,
    ReadState,
    SelectField,
    useHistory,
    useJobRun,
    WEATHER,
    type FanPoint,
} from '../parts'

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
    const run = useJobRun()
    useEffect(() => {
        const first = datasets.data?.[0]
        if (dataset === '' && first !== undefined) setDataset(first.id)
    }, [datasets.data, dataset])

    const start = () =>
        run.start(() =>
            api.backtest({
                dataset_id: dataset,
                ...(config === '' ? {} : { config_id: config }),
                ...(periods === null ? {} : { periods }),
                ...(splits === null ? {} : { splits }),
                ...(stride === null ? {} : { stride }),
                weather,
            }),
        )
    // From the form's fields too: the settings typed, then run.
    useShortcuts(
        [{ id: 'run-evaluation', key: 'Enter', mod: true, inFields: true, label: 'Run the evaluation', group: 'Evaluate', disabled: dataset === '', run: start }],
        [dataset, config, periods, splits, stride, weather, run.busy],
    )
    const noData = datasets.data !== undefined && datasets.data.length === 0
    return (
        <Section title="New evaluation" description="Backtest the model on a dataset: train on the past, forecast the next periods, and score against what happened, split after split.">
            <Stack gap="lg">
                <ReadState what="the datasets" resource={datasets} />
                <ReadState what="the configs" resource={configs} />
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
                    <Button onClick={start} disabled={dataset === ''} busy={run.busy}>
                        <Play aria-hidden />
                        Run evaluation
                    </Button>
                </div>
                {run.error && <Problem title={`The evaluation was refused: ${run.error.title}`} detail={run.error.detail} />}
                {run.running && (
                    <JobProgress
                        key={run.running.job}
                        jobId={run.running.job}
                        what="evaluation"
                        onDone={() => run.running && onDone(run.running.artifact)}
                        onEnd={run.onEnd}
                    />
                )}
            </Stack>
        </Section>
    )
}

type Past = Artifact<never, EvaluationSummary>

/** The columns of past runs, naming each one's dataset, from their summaries. */
const pastColumns = (names: ReadonlyMap<string, string>): DataColumn<Past>[] => [
    { id: 'created', header: 'Created', width: '11rem', cell: (row) => when(row.created_at) },
    {
        id: 'dataset',
        header: 'Dataset',
        width: 'minmax(10rem,1.5fr)',
        cell: (row) => {
            const id = row.data.summary?.dataset_id
            return id ? (names.get(id) ?? <span className="font-mono text-xs">{id}</span>) : <span className="text-muted-foreground">request data</span>
        },
    },
    {
        id: 'data',
        header: 'Data',
        width: 'minmax(10rem,1.5fr)',
        cell: (row) => {
            const data = row.data.summary?.data
            return <span className="font-mono text-xs">{data?.first ? `${data.first} to ${data.last ?? '-'}` : '-'}</span>
        },
    },
    {
        id: 'setup',
        header: 'Setup',
        width: 'minmax(10.5rem,1fr)',
        cell: (row) => {
            const options = row.data.summary?.options
            return options ? `${options.splits ?? '-'} splits x ${options.periods ?? '-'} periods` : '-'
        },
    },
    ...(['crps', 'mae', 'coverage_10_90'] as const).map(
        (id): DataColumn<Past> => ({
            id,
            header: metricName(id),
            width: '8.5rem',
            align: 'end',
            cell: (row) => <span className="tabular-nums">{score(row.data.summary?.aggregate[id], id.startsWith('coverage'))}</span>,
        }),
    ),
]

/** One forecast of a split, as its row in the values table. */
type ValueRow = BacktestForecast

const VALUE_COLUMNS: DataColumn<ValueRow>[] = [
    { id: 'location', header: 'Location', width: 'minmax(10rem,1.5fr)', cell: (row) => <span className="font-mono text-xs">{row.location}</span>, sortValue: (row) => row.location },
    { id: 'period', header: 'Period', width: 'minmax(7rem,1fr)', cell: (row) => <span className="font-mono text-xs">{row.time_period}</span>, sortValue: (row) => row.time_period },
    { id: 'horizon', header: 'Ahead', width: '5rem', align: 'end', cell: (row) => <span className="tabular-nums">{row.horizon}</span>, sortValue: (row) => row.horizon },
    ...(['observed', 'median', 'q10', 'q25', 'q75', 'q90'] as const).map(
        (key): DataColumn<ValueRow> => ({
            id: key,
            header: key === 'observed' ? 'Observed' : key === 'median' ? 'Median' : `${key.slice(1)}th`,
            width: '6.5rem',
            align: 'end',
            cell: (row) => <span className="tabular-nums">{score(row[key])}</span>,
            sortValue: (row) => row[key] ?? Number.NEGATIVE_INFINITY,
        }),
    ),
]

/** One evaluation's scores: tiles, by horizon, by location, and the forecasts against what happened. */
function EvaluationView({ id }: { id: string }) {
    const artifact = useResource((signal) => api.artifact<Evaluation>(id, signal), [id])
    const result = artifact.data?.data.content
    const datasetId = result?.dataset_id ?? undefined
    const dataset = useResource(datasetId === undefined ? null : (signal) => api.dataset(datasetId, signal), [datasetId])
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

    const shown = useMemo(() => forecasts.filter((row) => String(row.split) === split && chosen.includes(row.location)), [forecasts, split, chosen])
    const fans = useMemo(() => {
        const out = new Map<string, FanPoint[]>()
        for (const row of shown) out.set(row.location, [...(out.get(row.location) ?? []), { ...row, period: row.time_period }])
        return out
    }, [shown])
    // A split's forecasts start together; the service picks the history before them.
    const first = useMemo(() => shown.map((row) => row.time_period).sort()[0], [shown])
    const ahead = result?.options.periods ?? 1
    const history = useHistory(dataset.data, chosen, first, ahead)
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
        <PageState loading={artifact.loading} error={artifact.error?.detail ?? artifact.error?.message} status={artifact.error?.status} onRetry={artifact.reload}>
            {result && artifact.data && (
                <Stack gap="section">
                    <PageHeader
                        title="Evaluation"
                        code={id}
                        onBack={() => go('/evaluate')}
                        meta={`${when(artifact.data.created_at)}, ${dataset.data?.name ?? result.data.source ?? 'request data'}, ${count(result.rows)} scored forecasts in ${result.elapsed_ms.toFixed(1)} ms`}
                        actions={
                            <Button variant="outline" size="sm" onClick={() => void download(id)}>
                                <Download aria-hidden />
                                Download
                            </Button>
                        }
                    />
                    <DescriptionList
                        layout="row"
                        items={[
                            { term: 'Data', value: result.data.first ? `${result.data.first} to ${result.data.last ?? '-'}` : '-', code: true },
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
                            <Stack gap="lg">
                                <HistoryNote dataset={datasetId === undefined ? undefined : dataset} history={history} />
                                <div className="grid gap-4 lg:grid-cols-2">
                                    {chosen.map((location) => (
                                        <FanChart
                                            key={location}
                                            title={location}
                                            subtitle={`split ${Number(split) + 1}`}
                                            file={`evaluation-${location}`}
                                            points={[...(history.data?.get(location) ?? NONE), ...(fans.get(location) ?? NONE)]}
                                            loading={history.loading}
                                        />
                                    ))}
                                </div>
                            </Stack>
                        )}
                    </Section>
                    {forecasts.length > 0 && (
                        <Section title="Values" meta={`${count(shown.length)} forecasts of split ${Number(split) + 1}, for the locations chosen`} flush>
                            <DataTable
                                columns={VALUE_COLUMNS}
                                rows={shown}
                                rowKey={(row) => `${row.location}:${row.time_period}`}
                                framed={false}
                                bar={false}
                                paging="client"
                                defaultPageSize={25}
                                minWidth="56rem"
                                empty="Choose a location to see its forecasts."
                            />
                        </Section>
                    )}
                </Stack>
            )}
        </PageState>
    )
}

/** Downloads an evaluation's document, saying so when it cannot. */
const download = (id: string) =>
    api.downloadArtifact(id).catch((failure: Error) => toast.add({ type: 'error', title: 'Could not download the evaluation', description: failure.message }))

const PAGE = 25

/** The evaluations run before, newest first, a page at a time. */
function PastEvaluations() {
    const [page, setPage] = useState(0)
    const [size, setSize] = useState(PAGE)
    const history = useResource((signal) => api.evaluations(page + 1, size, signal), [page, size])
    const datasets = useResource(api.datasets, [])
    const names = useMemo(() => new Map((datasets.data ?? []).map((each) => [each.id, each.name])), [datasets.data])
    return (
        <Section title="Past evaluations" meta={history.data ? `${history.data.total}` : undefined} flush>
            {history.error ? (
                <ReadState what="the past evaluations" resource={history} />
            ) : (
                <DataTable
                    columns={pastColumns(names)}
                    rows={history.data?.items ?? []}
                    rowKey={(row) => row.id}
                    loading={history.loading}
                    paging="server"
                    total={history.data?.total}
                    page={page}
                    onPageChange={setPage}
                    pageSize={size}
                    onPageSizeChange={(next) => {
                        setSize(next)
                        setPage(0)
                    }}
                    pageSizes={[10, 25, 50, 100]}
                    onOpen={(row) => go(`/evaluate/${row.id}`)}
                    chevron
                    framed={false}
                    bar={false}
                    minWidth="48rem"
                    empty="No evaluation yet."
                />
            )}
        </Section>
    )
}

/** Evaluate: start a backtest, follow it, and read its scores; past evaluations below. */
export function Evaluate({ opened }: { opened: string | undefined }) {
    if (opened !== undefined) return <EvaluationView id={opened} />
    return (
        <Stack gap="section">
            <PageHeader title="Evaluate" description="How well the model forecasts, scored as chap-core scores it." />
            <NewEvaluation onDone={(artifactId) => go(`/evaluate/${artifactId}`)} />
            <PastEvaluations />
        </Stack>
    )
}
