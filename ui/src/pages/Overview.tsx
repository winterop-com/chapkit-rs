import { DashboardGrid, DashboardPanel } from '@winterop-com/sokkel-ui/dashboard/DashboardGrid'
import { StatPanel } from '@winterop-com/sokkel-ui/dashboard/StatPanel'
import { DescriptionList } from '@winterop-com/sokkel-ui/DescriptionList'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { Stat } from '@winterop-com/sokkel-ui/Stat'
import { StatusChip } from '@winterop-com/sokkel-ui/StatusChip'
import { TextLink } from '@winterop-com/sokkel-ui/TextLink'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'

import { api } from '../api/client'
import { useModel } from '../App'
import { METRICS, periodWord, score, when } from '../lib/format'
import { useResource } from '../lib/resource'
import { go } from '../lib/router'

const HEADLINE = METRICS.slice(0, 4)

/** The front page: the model, what the service holds, and how the latest evaluation scored. */
export function Overview() {
    const model = useModel()
    const health = useResource(api.health, [], 15_000)
    const datasets = useResource(api.datasets, [])
    const configs = useResource(api.configs, [])
    // The newest evaluation's summary, and with it how many there are.
    const evaluations = useResource((signal) => api.evaluations(1, 1, signal), [])
    const forecasts = useResource((signal) => api.forecasts(1, 1, signal), [])
    const latest = evaluations.data?.items[0]
    const result = latest?.data.summary
    const meta = model?.model_metadata

    return (
        <>
            <PageHeader
                title={model?.display_name ?? 'Model'}
                code={model?.id}
                description={model?.description ?? undefined}
                lead={health.error !== undefined ? <StatusChip status="unreachable" /> : health.data !== undefined ? <StatusChip status={health.data.status} /> : undefined}
                actions={
                    <>
                        <Button variant="outline" size="sm" onClick={() => go('/forecast')}>
                            Forecast
                        </Button>
                        <Button size="sm" onClick={() => go('/evaluate')}>
                            Evaluate
                        </Button>
                    </>
                }
            />
            <DashboardGrid>
                <StatPanel
                    title="Datasets"
                    value={datasets.data?.length ?? null}
                    loading={datasets.loading}
                    error={datasets.error?.message}
                    onRetry={datasets.reload}
                    href="#/datasets"
                    span="quarter"
                />
                <StatPanel
                    title="Evaluations"
                    value={evaluations.data?.total ?? null}
                    loading={evaluations.loading}
                    error={evaluations.error?.message}
                    onRetry={evaluations.reload}
                    href="#/evaluate"
                    span="quarter"
                />
                <StatPanel
                    title="Forecasts"
                    value={forecasts.data?.total ?? null}
                    loading={forecasts.loading}
                    error={forecasts.error?.message}
                    onRetry={forecasts.reload}
                    href="#/forecast"
                    span="quarter"
                />
                <StatPanel
                    title="Configs"
                    value={configs.data?.length ?? null}
                    loading={configs.loading}
                    error={configs.error?.message}
                    onRetry={configs.reload}
                    href="#/configs"
                    span="quarter"
                />
                <DashboardPanel title="Model" subtitle={model ? `version ${model.version}` : undefined} span="half" loading={model === undefined}>
                    <DescriptionList
                        termWidth="10rem"
                        items={[
                            { term: 'Version', value: model?.version, code: true },
                            { term: 'chapkit-rs', value: model?.chapkit_rs_version, code: true },
                            { term: 'Period type', value: periodWord(model?.period_type) },
                            { term: 'Target', value: model?.target, code: true },
                            {
                                term: 'Covariates',
                                value: model === undefined ? undefined : model.required_covariates.length === 0 ? 'None required' : model.required_covariates.join(', '),
                            },
                            {
                                term: 'Status',
                                value: meta?.author_assessed_status ? <StatusChip status={meta.author_assessed_status} /> : undefined,
                            },
                            { term: 'Author', value: meta?.author },
                            {
                                term: 'Documentation',
                                value: meta?.documentation_url ? (
                                    <TextLink href={meta.documentation_url} external>
                                        {meta.documentation_url.replace(/^https?:\/\//, '')}
                                    </TextLink>
                                ) : undefined,
                            },
                        ]}
                    />
                    {meta?.author_note && <p className="mt-3 text-sm text-muted-foreground">{meta.author_note}</p>}
                </DashboardPanel>
                <DashboardPanel
                    title="Latest evaluation"
                    subtitle={latest ? when(latest.created_at) : undefined}
                    span="half"
                    loading={evaluations.loading}
                    error={evaluations.error?.message}
                    onRetry={evaluations.reload}
                    empty={!evaluations.loading && evaluations.error === undefined && latest === undefined}
                    words={{ empty: 'No evaluation yet. Evaluate the model on a dataset to see how it scores.' }}
                    actions={
                        latest && (
                            <Button size="xs" variant="ghost" onClick={() => go(`/evaluate/${latest.id}`)}>
                                Open
                            </Button>
                        )
                    }
                >
                    {result && (
                        <div className="flex flex-col gap-4">
                            <div className="grid grid-cols-2 gap-4">
                                {HEADLINE.map((metric) => (
                                    <Stat key={metric.id} label={metric.name} value={score(result.aggregate[metric.id], metric.share)} />
                                ))}
                            </div>
                            <DescriptionList
                                layout="row"
                                items={[
                                    { term: 'Splits', value: result.options.splits, numeric: true },
                                    { term: 'Periods', value: result.options.periods, numeric: true },
                                    { term: 'Locations', value: result.data.locations, numeric: true },
                                    { term: 'Range', value: result.data.first ? `${result.data.first} to ${result.data.last ?? '-'}` : undefined },
                                ]}
                            />
                        </div>
                    )}
                </DashboardPanel>
            </DashboardGrid>
        </>
    )
}
