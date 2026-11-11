import { DataTable, type DataColumn } from '@winterop-com/sokkel-ui/DataTable'
import { DescriptionList } from '@winterop-com/sokkel-ui/DescriptionList'
import { useCommands } from '@winterop-com/sokkel-ui/hooks/use-commands'
import { useShortcuts } from '@winterop-com/sokkel-ui/hooks/use-shortcuts'
import { keyLabels } from '@winterop-com/sokkel-ui/lib/keys'
import { ConfirmDialog, DialogShell } from '@winterop-com/sokkel-ui/DialogShell'
import { PageHeader } from '@winterop-com/sokkel-ui/PageHeader'
import { Problem } from '@winterop-com/sokkel-ui/Problem'
import { SchemaForm, SchemaIssueCount, useSchemaForm } from '@winterop-com/sokkel-ui/schema/SchemaForm'
import { TextField } from '@winterop-com/sokkel-ui/TextField'
import { Button } from '@winterop-com/sokkel-ui/primitives/button'
import { toast } from '@winterop-com/sokkel-ui/primitives/toast'
import { Plus } from 'lucide-react'
import { useState } from 'react'

import { api, asApiError } from '../api/client'
import type { Config } from '../api/types'
import { ACTIONS, NEW_CONFIG } from '../lib/commands'
import { dateOf, when } from '../lib/format'
import { go, hashParam } from '../lib/router'
import { useResource } from '../lib/resource'
import { ReadState } from '../parts'

const said = (value: unknown) => (typeof value === 'string' ? value : JSON.stringify(value))

const COLUMNS: DataColumn<Config>[] = [
    { id: 'name', header: 'Name', width: 'minmax(10rem,1fr)', cell: (row) => <span className="font-medium">{row.name}</span>, sortValue: (row) => row.name },
    {
        id: 'options',
        header: 'Options',
        width: 'minmax(16rem,2fr)',
        cell: (row) => (
            <span className="truncate font-mono text-xs text-muted-foreground">
                {Object.entries(row.data)
                    .map(([key, value]) => `${key}=${said(value)}`)
                    .join('  ')}
            </span>
        ),
    },
    { id: 'created', header: 'Created', width: '11rem', cell: (row) => when(row.created_at), sortValue: (row) => dateOf(row.created_at).getTime() },
]

/** A new config, its options drawn from the model's JSON schema. */
function NewConfig({ schema, onClose, onSaved }: { schema: Record<string, unknown>; onClose: () => void; onSaved: () => void }) {
    const form = useSchemaForm(schema)
    const [name, setName] = useState('')
    const [busy, setBusy] = useState(false)
    const [error, setError] = useState<{ title: string; detail?: string }>()
    const save = () => {
        if (!form.submit()) return
        setBusy(true)
        api.createConfig(name.trim(), form.value)
            .then(() => {
                toast.add({ type: 'success', title: `${name.trim()} saved` })
                onSaved()
            })
            .catch((failure: unknown) => {
                const refused = asApiError(failure)
                setError({ title: refused.message, detail: refused.detail })
            })
            .finally(() => setBusy(false))
    }
    return (
        <DialogShell
            open
            onOpenChange={(open) => !open && onClose()}
            width="lg"
            title="New config"
            description="A named set of the model's options, to evaluate and forecast with."
            count={<SchemaIssueCount form={form} />}
            confirm={{ label: 'Save', onConfirm: save, busy, why: name.trim() === '' ? 'Name it first' : undefined }}
        >
            <div className="flex flex-col gap-4">
                <TextField id="config-name" label="Name" required value={name} onChange={setName} placeholder="three-months" />
                <SchemaForm form={form} fallbacks />
                {error && <Problem title={`The config was refused: ${error.title}`} detail={error.detail} />}
            </div>
        </DialogShell>
    )
}

/** Configs: the named option sets, made from the model's schema. */
export function Configs({ opened }: { opened?: string }) {
    const configs = useResource(api.configs, [])
    const schema = useResource(api.configSchema, [])
    // `#/configs?new` (the palette's New config from another page) opens the form.
    const [creating, setCreating] = useState(() => hashParam('new') !== undefined)
    const stopCreating = () => {
        setCreating(false)
        if (hashParam('new') !== undefined) go('/configs')
    }
    useShortcuts([{ id: NEW_CONFIG, key: 'n', label: 'New config', group: 'Configs', run: () => setCreating(true) }], [])
    useCommands([{ title: ACTIONS, rank: 10, commands: [{ id: NEW_CONFIG, label: 'New config', icon: Plus, keys: keyLabels({ key: 'n' }), onSelect: () => setCreating(true) }] }], [])
    const [removing, setRemoving] = useState<Config | null>(null)
    const remove = (config: Config) =>
        api.deleteConfig(config.id).then(
            () => {
                toast.add({ title: `${config.name} deleted` })
                setRemoving(null)
                configs.reload()
            },
            (failure: Error) => toast.add({ type: 'error', title: 'Could not delete the config', description: failure.message }),
        )
    return (
        <>
            <PageHeader
                title="Configs"
                count={configs.data?.length}
                description="Named sets of the model's options, to evaluate and forecast with."
                actions={
                    <Button
                        size="sm"
                        onClick={() => setCreating(true)}
                        disabled={schema.data === undefined}
                        reason={schema.error !== undefined ? "Could not read the model's options" : schema.loading ? "Reading the model's options" : undefined}
                    >
                        <Plus aria-hidden />
                        New config
                    </Button>
                }
            />
            <ReadState what="the model's options" resource={schema} tone="warning" />
            {configs.error ? <ReadState what="the configs" resource={configs} /> : <DataTable
                columns={COLUMNS}
                rows={configs.data ?? []}
                rowKey={(row) => row.id}
                loading={configs.loading}
                defaultSort={{ column: 'created', direction: 'desc' }}
                minWidth="40rem"
                defaultExpanded={opened === undefined ? undefined : new Set([opened])}
                detail={(row) => (
                    <DescriptionList
                        termWidth="14rem"
                        items={[
                            { term: 'Id', value: row.id, code: true },
                            ...Object.entries(row.data).map(([key, value]) => ({ term: key, value: said(value), code: true })),
                        ]}
                    />
                )}
                rowActions={(row) => [{ id: 'delete', label: 'Delete', destructive: true, onSelect: () => setRemoving(row) }]}
                empty="No config yet. Evaluations and forecasts use the model defaults until you make one."
            />}
            {creating && schema.data && (
                <NewConfig
                    schema={schema.data}
                    onClose={stopCreating}
                    onSaved={() => {
                        stopCreating()
                        configs.reload()
                    }}
                />
            )}
            <ConfirmDialog
                open={removing !== null}
                onOpenChange={(open) => !open && setRemoving(null)}
                question={`Delete ${removing?.name ?? 'the config'}?`}
                confirm={{ label: 'Delete', destructive: true, onConfirm: () => removing && void remove(removing) }}
            />
        </>
    )
}
