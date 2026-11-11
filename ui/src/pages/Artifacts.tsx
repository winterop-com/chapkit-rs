import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { DescriptionList } from '@winterop-com/sokkel-ui/DescriptionList'
import { ConfirmDialog } from '@winterop-com/sokkel-ui/DialogShell'
import { JsonTree } from '@winterop-com/sokkel-ui/JsonTree'
import { Stack } from '@winterop-com/sokkel-ui/Layout'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { PageState } from '@winterop-com/sokkel-ui/PageState'
import { Section } from '@winterop-com/sokkel-ui/Section'
import { Badge } from '@winterop-com/sokkel-ui/primitives/badge'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { toast } from '@winterop-com/sokkel-ui/primitives/toast'
import { Download } from 'lucide-react'
import { useMemo, useState } from 'react'

import { api } from '../api/client'
import type { Artifact } from '../api/types'
import { bytes, dateOf, when } from '../lib/format'
import { useResource } from '../lib/resource'
import { go } from '../lib/router'
import { ReadState } from '../parts'

const TYPE_WORDS: Record<string, string> = {
    ml_backtest: 'Evaluation',
    ml_forecast: 'Forecast',
    ml_training: 'Trained model',
    ml_prediction: 'Prediction',
}

/** Where an artifact is read best: its own page for evaluations and forecasts, else its detail here. */
const pageOf = (artifact: Artifact) =>
    artifact.data.type === 'ml_backtest'
        ? `/evaluate/${artifact.id}`
        : artifact.data.type === 'ml_forecast'
          ? `/forecast/${artifact.id}`
          : `/artifacts/${artifact.id}`

/** The size of an artifact's content: as stored, or as its JSON when the service does not say (a listing carries neither). */
function sizeOf(artifact: Artifact): number | undefined {
    if (artifact.data.content_size !== null && artifact.data.content_size !== undefined) return artifact.data.content_size
    return artifact.data.content === undefined ? undefined : JSON.stringify(artifact.data.content).length
}

const COLUMNS: DataColumn<Artifact>[] = [
    { id: 'id', header: 'Artifact', width: 'minmax(16rem,1.5fr)', cell: (row) => <span className="font-mono text-xs">{row.id}</span> },
    {
        id: 'type',
        header: 'Type',
        width: '10rem',
        cell: (row) => (row.data.type ? <Badge variant="outline">{TYPE_WORDS[row.data.type] ?? row.data.type}</Badge> : '-'),
        sortValue: (row) => row.data.type ?? '',
    },
    {
        id: 'level',
        header: 'Level',
        width: '5rem',
        align: 'end',
        cell: (row) => <span className="tabular-nums">{row.level}</span>,
        sortValue: (row) => row.level,
    },
    { id: 'created', header: 'Created', width: '11rem', cell: (row) => when(row.created_at), sortValue: (row) => dateOf(row.created_at).getTime() },
]

/** Downloads an artifact's content, saying so when it cannot. */
const download = (id: string) =>
    api.downloadArtifact(id).catch((failure: Error) => toast.add({ type: 'error', title: 'Could not download the artifact', description: failure.message }))

function ArtifactView({ id }: { id: string }) {
    const artifact = useResource((signal) => api.artifact(id, signal), [id])
    const data = artifact.data
    return (
        <PageState loading={artifact.loading} error={artifact.error?.detail ?? artifact.error?.message} status={artifact.error?.status} onRetry={artifact.reload}>
            {data && (
                <Stack gap="section">
                    <PageHeader
                        title={TYPE_WORDS[data.data.type ?? ''] ?? data.data.type ?? 'Artifact'}
                        code={data.id}
                        onBack={() => go('/artifacts')}
                        actions={
                            <Button size="sm" variant="outline" onClick={() => void download(id)}>
                                <Download aria-hidden />
                                Download
                            </Button>
                        }
                    />
                    <Section title="Facts">
                        <DescriptionList
                            columns={2}
                            items={[
                                { term: 'Type', value: data.data.type, code: true },
                                { term: 'Content type', value: data.data.content_type, code: true },
                                { term: 'Size', value: bytes(sizeOf(data)), numeric: true },
                                { term: 'Level', value: data.level, numeric: true },
                                { term: 'Parent', value: data.parent_id, code: true },
                                { term: 'Created', value: when(data.created_at) },
                            ]}
                        />
                    </Section>
                    {data.data.metadata && (
                        <Section title="Metadata">
                            <JsonTree value={data.data.metadata} open={2} />
                        </Section>
                    )}
                    <Section title="Content">
                        <JsonTree value={data.data.content ?? null} open={1} limit={50} />
                    </Section>
                </Stack>
            )}
        </PageState>
    )
}

/** Artifacts: everything the service stored, as the tree of what came from what. */
export function Artifacts({ opened }: { opened: string | undefined }) {
    const artifacts = useResource((signal) => api.artifacts(undefined, signal), [])
    const [removing, setRemoving] = useState<Artifact | null>(null)
    const { roots, children } = useMemo(() => {
        const all = artifacts.data ?? []
        const ids = new Set(all.map((each) => each.id))
        const byParent = new Map<string, Artifact[]>()
        for (const each of all) {
            if (each.parent_id && ids.has(each.parent_id)) byParent.set(each.parent_id, [...(byParent.get(each.parent_id) ?? []), each])
        }
        return { roots: all.filter((each) => !each.parent_id || !ids.has(each.parent_id)), children: byParent }
    }, [artifacts.data])
    if (opened !== undefined) return <ArtifactView id={opened} />
    const remove = (artifact: Artifact) =>
        api.deleteArtifact(artifact.id).then(
            () => {
                toast.add({ title: 'Artifact deleted' })
                setRemoving(null)
                artifacts.reload()
            },
            (failure: Error) => toast.add({ type: 'error', title: 'Could not delete the artifact', description: failure.message }),
        )
    return (
        <>
            <PageHeader title="Artifacts" count={artifacts.data?.length} description="What the service stored: evaluations, forecasts, trained models and predictions." />
            {artifacts.error ? <ReadState what="the artifacts" resource={artifacts} /> : <DataTable
                columns={COLUMNS}
                rows={roots}
                rowKey={(row) => row.id}
                childRows={(row) => children.get(row.id)}
                loading={artifacts.loading}
                defaultSort={{ column: 'created', direction: 'desc' }}
                onOpen={(row) => go(pageOf(row))}
                chevron
                minWidth="44rem"
                paging="client"
                defaultPageSize={50}
                rowActions={(row) => [
                    { id: 'open', label: 'Open', onSelect: () => go(pageOf(row)) },
                    { id: 'details', label: 'Details', onSelect: () => go(`/artifacts/${row.id}`) },
                    { id: 'download', label: 'Download', onSelect: () => void download(row.id) },
                    { id: 'sep', separator: true },
                    { id: 'delete', label: 'Delete', destructive: true, onSelect: () => setRemoving(row) },
                ]}
                empty="Nothing stored yet. Evaluations and forecasts are kept here."
            />}
            <ConfirmDialog
                open={removing !== null}
                onOpenChange={(open) => !open && setRemoving(null)}
                question="Delete this artifact?"
                detail={removing ? `${TYPE_WORDS[removing.data.type ?? ''] ?? 'Artifact'} ${removing.id}, and the artifacts made from it.` : undefined}
                confirm={{ label: 'Delete', destructive: true, onConfirm: () => removing && void remove(removing) }}
            />
        </>
    )
}
