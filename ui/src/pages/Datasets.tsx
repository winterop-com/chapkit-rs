import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { ConfirmDialog, DialogShell } from '@winterop-com/sokkel-ui/DialogShell'
import { FileField } from '@winterop-com/sokkel-ui/FileField'
import { Stack } from '@winterop-com/sokkel-ui/Layout'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { Problem } from '@winterop-com/sokkel-ui/Problem'
import { TextField } from '@winterop-com/sokkel-ui/TextField'
import { DropZone } from '@winterop-com/sokkel-ui/UploadTray'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from '@winterop-com/sokkel-ui/primitives/empty'
import { toast } from '@winterop-com/sokkel-ui/primitives/toast'
import { Upload } from 'lucide-react'
import { useState } from 'react'

import { api } from '../api/client'
import type { Dataset } from '../api/types'
import { count, dateOf, periodWord, when } from '../lib/format'
import { useResource } from '../lib/resource'
import { go } from '../lib/router'

const COLUMNS: DataColumn<Dataset>[] = [
    { id: 'name', header: 'Name', width: 'minmax(12rem,2fr)', cell: (row) => <span className="font-medium">{row.name}</span>, sortValue: (row) => row.name },
    { id: 'period', header: 'Period type', width: '7rem', cell: (row) => periodWord(row.period_type), sortValue: (row) => row.period_type },
    {
        id: 'range',
        header: 'Range',
        width: 'minmax(12rem,1.5fr)',
        cell: (row) => (
            <span className="font-mono text-xs">
                {row.first_period ?? '-'} to {row.last_period ?? '-'}
            </span>
        ),
        sortValue: (row) => row.first_period ?? '',
    },
    { id: 'locations', header: 'Locations', width: '6rem', align: 'end', cell: (row) => count(row.locations), sortValue: (row) => row.locations },
    { id: 'rows', header: 'Rows', width: '6rem', align: 'end', cell: (row) => count(row.rows), sortValue: (row) => row.rows },
    {
        id: 'covariates',
        header: 'Covariates',
        width: 'minmax(10rem,1.5fr)',
        cell: (row) => <span className="truncate text-muted-foreground">{row.covariates.length === 0 ? 'None' : row.covariates.join(', ')}</span>,
    },
    { id: 'created', header: 'Created', width: '10rem', cell: (row) => when(row.created_at), sortValue: (row) => dateOf(row.created_at).getTime() },
]

/** A file's name without its extension, as a dataset's first name. */
const baseName = (file: File) => file.name.replace(/\.[^.]+$/, '')

/** Takes a CSV or JSON file and a name, and stores it as a dataset. */
export function UploadDialog({
    open,
    onOpenChange,
    initial,
    onUploaded,
}: {
    open: boolean
    onOpenChange: (open: boolean) => void
    initial: File[]
    onUploaded: (dataset: Dataset) => void
}) {
    const [files, setFiles] = useState<File[]>(initial)
    const [name, setName] = useState(initial[0] ? baseName(initial[0]) : '')
    const [busy, setBusy] = useState(false)
    const [error, setError] = useState<{ title: string; detail?: string }>()
    const file = files[0]
    const upload = () => {
        if (file === undefined) return
        setBusy(true)
        setError(undefined)
        api.uploadDataset(name.trim() || baseName(file), file)
            .then((dataset) => {
                toast.add({ type: 'success', title: `${dataset.name} uploaded`, description: `${count(dataset.rows)} rows, ${dataset.locations} locations` })
                onUploaded(dataset)
            })
            .catch((failure: { message: string; detail?: string }) => setError({ title: failure.message, detail: failure.detail }))
            .finally(() => setBusy(false))
    }
    return (
        <DialogShell
            open={open}
            onOpenChange={onOpenChange}
            title="Upload a dataset"
            description="A CSV or JSON table with time_period, location, the target and any covariates."
            confirm={{
                label: 'Upload',
                onConfirm: upload,
                busy,
                busyLabel: 'Uploading',
                why: file === undefined ? 'Choose a file first' : undefined,
            }}
        >
            <div className="flex flex-col gap-4">
                <FileField
                    id="dataset-file"
                    label="File"
                    accept={['.csv', '.json']}
                    maxFiles={1}
                    previews={false}
                    value={files}
                    onChange={(next) => {
                        setFiles(next)
                        if (next[0] && name === '') setName(baseName(next[0]))
                    }}
                />
                <TextField id="dataset-name" label="Name" value={name} onChange={setName} placeholder={file ? baseName(file) : 'laos-monthly'} />
                {error && <Problem title={`The upload was refused: ${error.title}`} detail={error.detail} />}
            </div>
        </DialogShell>
    )
}

/** The datasets the service holds, an upload by drop or picker, and a way into each. */
export function Datasets() {
    const datasets = useResource(api.datasets, [])
    const [upload, setUpload] = useState<File[] | null>(null)
    const [removing, setRemoving] = useState<Dataset | null>(null)
    const rows = datasets.data ?? []

    const remove = (dataset: Dataset) =>
        api.deleteDataset(dataset.id).then(
            () => {
                toast.add({ title: `${dataset.name} deleted` })
                setRemoving(null)
                datasets.reload()
            },
            (failure: Error) => toast.add({ type: 'error', title: 'Could not delete the dataset', description: failure.message }),
        )

    return (
        <DropZone onFiles={(files) => setUpload(files.slice(0, 1))} words={{ title: 'Drop to upload', detail: 'A CSV or JSON dataset' }}>
            <Stack gap="section">
            <PageHeader
                title="Datasets"
                count={datasets.data ? rows.length : undefined}
                description="Data to evaluate the model on and to forecast from. Drop a CSV or JSON file anywhere here to upload it."
                actions={
                    <Button size="sm" onClick={() => setUpload([])}>
                        <Upload aria-hidden />
                        Upload
                    </Button>
                }
            />
            {datasets.error ? (
                <Problem title="Could not read the datasets" detail={datasets.error.detail} technical={datasets.error.message} actions={<Button size="sm" variant="outline" onClick={datasets.reload}>Try again</Button>} />
            ) : (
                <DataTable
                    columns={COLUMNS}
                    rows={rows}
                    rowKey={(row) => row.id}
                    loading={datasets.loading}
                    defaultSort={{ column: 'created', direction: 'desc' }}
                    onOpen={(row) => go(`/datasets/${row.id}`)}
                    chevron
                    minWidth="56rem"
                    rowActions={(row) => [
                        { id: 'open', label: 'Open', onSelect: () => go(`/datasets/${row.id}`) },
                        { id: 'evaluate', label: 'Evaluate on it', onSelect: () => go(`/evaluate?dataset=${row.id}`) },
                        { id: 'forecast', label: 'Forecast from it', onSelect: () => go(`/forecast?dataset=${row.id}`) },
                        { id: 'sep', separator: true },
                        { id: 'delete', label: 'Delete', destructive: true, onSelect: () => setRemoving(row) },
                    ]}
                    empty={
                        <Empty>
                            <EmptyHeader>
                                <EmptyTitle>No datasets yet</EmptyTitle>
                                <EmptyDescription>Upload a CSV or JSON file, or drop one on this page.</EmptyDescription>
                            </EmptyHeader>
                            <EmptyContent>
                                <Button size="sm" variant="outline" onClick={() => setUpload([])}>
                                    Upload a dataset
                                </Button>
                            </EmptyContent>
                        </Empty>
                    }
                />
            )}
            </Stack>
            {upload !== null && (
                <UploadDialog
                    open
                    initial={upload}
                    onOpenChange={(open) => !open && setUpload(null)}
                    onUploaded={(dataset) => {
                        setUpload(null)
                        datasets.reload()
                        go(`/datasets/${dataset.id}`)
                    }}
                />
            )}
            <ConfirmDialog
                open={removing !== null}
                onOpenChange={(open) => !open && setRemoving(null)}
                question={`Delete ${removing?.name ?? 'the dataset'}?`}
                detail="Evaluations and forecasts made from it are kept."
                confirm={{ label: 'Delete', destructive: true, onConfirm: () => removing && void remove(removing) }}
            />
        </DropZone>
    )
}
