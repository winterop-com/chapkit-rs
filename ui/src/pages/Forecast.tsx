import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { DescriptionList } from '@winterop-com/sokkel-ui/DescriptionList'
import { Stack } from '@winterop-com/sokkel-ui/Layout'
import { NumberField } from '@winterop-com/sokkel-ui/NumberField'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { PageState } from '@winterop-com/sokkel-ui/PageState'
import { Problem } from '@winterop-com/sokkel-ui/Problem'
import { Section } from '@winterop-com/sokkel-ui/Section'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { Download, Play } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { api } from '../api/client'
import type { Artifact, Forecast as ForecastContent, ForecastRow } from '../api/types'
import { count, dateOf, score, when } from '../lib/format'
import { saveFile, toCsv } from '../lib/csv'
import { useResource } from '../lib/resource'
import { go, hashParam } from '../lib/router'
import { configOptions, datasetOptions, JobProgress, LocationFan, LocationPicker, SelectField, WEATHER, type FanPoint } from '../parts'

function NewForecast({ onDone }: { onDone: (artifactId: string) => void }) {
    const datasets = useResource(api.datasets, [])
    const configs = useResource(api.configs, [])
    const [dataset, setDataset] = useState(hashParam('dataset') ?? '')
    const [config, setConfig] = useState('')
    const [periods, setPeriods] = useState<number | null>(3)
    const [weather, setWeather] = useState('climatology')
    const [running, setRunning] = useState<{ job: string; artifact: string }>()
    const [error, setError] = useState<{ title: string; detail?: string }>()
    useEffect(() => {
        const first = datasets.data?.[0]
        if (dataset === '' && first !== undefined) setDataset(first.id)
    }, [datasets.data, dataset])
    const start = () => {
        setError(undefined)
        api.forecast({ dataset_id: dataset, ...(config === '' ? {} : { config_id: config }), ...(periods === null ? {} : { periods }), weather }).then(
            (submitted) => setRunning({ job: submitted.job_id, artifact: submitted.artifact_id }),
            (failure: { message: string; detail?: string }) => setError({ title: failure.message, detail: failure.detail }),
        )
    }
    const noData = datasets.data !== undefined && datasets.data.length === 0
    return (
        <Section title="New forecast" description="Train on the whole dataset and forecast the periods after its last one.">
            <Stack gap="lg">
                {noData ? (
                    <Problem
                        tone="warning"
                        title="There is no dataset to forecast from"
                        detail="Upload one first."
                        actions={
                            <Button size="sm" variant="outline" onClick={() => go('/datasets')}>
                                Go to datasets
                            </Button>
                        }
                    />
                ) : (
                    <div className="grid gap-4 sm:grid-cols-2">
                        <SelectField id="fc-dataset" label="Dataset" value={dataset} options={datasetOptions(datasets.data)} onChange={setDataset} />
                        <SelectField id="fc-config" label="Config" value={config} placeholder="Model defaults" options={configOptions(configs.data)} onChange={setConfig} />
                        <NumberField id="fc-periods" label="Periods" hint="ahead" integer min={1} value={periods} onValueChange={setPeriods} />
                        <SelectField id="fc-weather" label="Future weather" value={weather} options={WEATHER.slice(0, 2)} onChange={setWeather} />
                    </div>
                )}
                <div>
                    <Button onClick={start} disabled={dataset === '' || (running !== undefined && error === undefined)}>
                        <Play aria-hidden />
                        Run forecast
                    </Button>
                </div>
                {error && <Problem title={`The forecast was refused: ${error.title}`} detail={error.detail} />}
                {running && <JobProgress key={running.job} jobId={running.job} what="forecast" onDone={() => onDone(running.artifact)} />}
            </Stack>
        </Section>
    )
}

const ROW_COLUMNS: DataColumn<ForecastRow>[] = [
    { id: 'location', header: 'Location', width: 'minmax(10rem,1.5fr)', cell: (row) => <span className="font-mono text-xs">{row.location}</span>, sortValue: (row) => row.location },
    { id: 'period', header: 'Period', width: 'minmax(8rem,1fr)', cell: (row) => <span className="font-mono text-xs">{row.time_period}</span>, sortValue: (row) => row.time_period },
    ...(['median', 'q10', 'q25', 'q75', 'q90'] as const).map(
        (key): DataColumn<ForecastRow> => ({
            id: key,
            header: key === 'median' ? 'Median' : `${key.slice(1)}th`,
            width: '6.5rem',
            align: 'end',
            cell: (row) => <span className="tabular-nums">{score(row[key])}</span>,
            sortValue: (row) => row[key] ?? 0,
        }),
    ),
]

function ForecastView({ id }: { id: string }) {
    const artifact = useResource(() => api.artifact<ForecastContent>(id), [id])
    const result = artifact.data?.data.content
    const dataset = useResource(result ? () => api.dataset(result.dataset_id) : null, [result?.dataset_id])
    const rows = useMemo(() => result?.forecasts ?? [], [result])
    const locations = useMemo(() => [...new Set(rows.map((row) => row.location))], [rows])
    const [chosen, setChosen] = useState<string[]>([])
    useEffect(() => setChosen(locations.slice(0, 4)), [locations])
    const fans = useMemo(() => {
        const out = new Map<string, FanPoint[]>()
        for (const row of rows) out.set(row.location, [...(out.get(row.location) ?? []), { ...row, period: row.time_period }])
        return out
    }, [rows])
    const download = () =>
        saveFile(`forecast-${dataset.data?.name ?? id}.csv`, toCsv(rows, ['location', 'time_period', 'median', 'q10', 'q25', 'q75', 'q90']))

    return (
        <PageState loading={artifact.loading} error={artifact.error?.detail ?? artifact.error?.message} status={artifact.error?.status}>
            {result && artifact.data && (
                <Stack gap="section">
                    <PageHeader
                        title="Forecast"
                        code={id}
                        onBack={() => go('/forecast')}
                        meta={`${when(artifact.data.created_at)}, from ${dataset.data?.name ?? result.dataset_id}`}
                        actions={
                            <Button size="sm" onClick={download}>
                                <Download aria-hidden />
                                Download CSV
                            </Button>
                        }
                    />
                    <DescriptionList
                        layout="row"
                        items={[
                            { term: 'Dataset', value: dataset.data?.name ?? result.dataset_id },
                            { term: 'Periods', value: result.periods, numeric: true },
                            { term: 'Locations', value: locations.length, numeric: true },
                            { term: 'Weather', value: result.weather ?? '-' },
                            { term: 'From', value: rows[0]?.time_period, code: true },
                        ]}
                    />
                    <Section
                        title="Forecast by location"
                        description="The observed history, then the median forecast with its 10-90 and 25-75 intervals."
                        actions={<LocationPicker locations={locations} value={chosen} onChange={setChosen} />}
                    >
                        <div className="grid gap-4 lg:grid-cols-2">
                            {chosen.map((location) => (
                                <LocationFan
                                    key={location}
                                    dataset={dataset.data}
                                    location={location}
                                    forecast={fans.get(location) ?? []}
                                    subtitle="recent history and the forecast"
                                    file={`forecast-${location}`}
                                />
                            ))}
                        </div>
                    </Section>
                    <Section title="Values" meta={`${count(rows.length)} rows`} flush>
                        <DataTable
                            columns={ROW_COLUMNS}
                            rows={rows}
                            rowKey={(row) => `${row.location}:${row.time_period}`}
                            framed={false}
                            bar={false}
                            paging="client"
                            defaultPageSize={25}
                            minWidth="48rem"
                        />
                    </Section>
                </Stack>
            )}
        </PageState>
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
            const id = (row.data.content as ForecastContent | undefined)?.dataset_id
            return id ? (names.get(id) ?? <span className="font-mono text-xs">{id}</span>) : <span className="text-muted-foreground">request data</span>
        },
    },
    {
        id: 'periods',
        header: 'Periods',
        width: '6rem',
        align: 'end',
        cell: (row) => (row.data.content as ForecastContent | undefined)?.periods ?? '-',
    },
    {
        id: 'from',
        header: 'From',
        width: 'minmax(8rem,1fr)',
        cell: (row) => <span className="font-mono text-xs">{(row.data.content as ForecastContent | undefined)?.forecasts[0]?.time_period ?? '-'}</span>,
    },
    {
        id: 'locations',
        header: 'Locations',
        width: '7rem',
        align: 'end',
        cell: (row) => new Set((row.data.content as ForecastContent | undefined)?.forecasts.map((each) => each.location)).size,
    },
]

/** Forecast: start one, follow it, and read it as fans and a table; past forecasts below. */
export function Forecast({ opened }: { opened: string | undefined }) {
    const history = useResource(() => api.recent('ml_forecast', 50), [opened])
    const datasets = useResource(api.datasets, [])
    const names = useMemo(() => new Map((datasets.data ?? []).map((each) => [each.id, each.name])), [datasets.data])
    if (opened !== undefined) return <ForecastView id={opened} />
    return (
        <Stack gap="section">
            <PageHeader title="Forecast" description="Forecast the periods after a dataset's last one, with intervals." />
            <NewForecast onDone={(artifactId) => go(`/forecast/${artifactId}`)} />
            <Section title="Past forecasts" meta={history.data ? `${history.data.length}` : undefined} flush>
                <DataTable
                    columns={pastColumns(names)}
                    rows={history.data ?? []}
                    rowKey={(row) => row.id}
                    loading={history.loading}
                    defaultSort={{ column: 'created', direction: 'desc' }}
                    onOpen={(row) => go(`/forecast/${row.id}`)}
                    chevron
                    framed={false}
                    bar={false}
                    minWidth="36rem"
                    empty="No forecast yet."
                />
            </Section>
        </Stack>
    )
}
