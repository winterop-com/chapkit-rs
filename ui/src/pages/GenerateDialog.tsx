import { DialogShell } from '@winterop-com/sokkel-ui/DialogShell'
import { NumberField } from '@winterop-com/sokkel-ui/NumberField'
import { Problem } from '@winterop-com/sokkel-ui/Problem'
import { TextField } from '@winterop-com/sokkel-ui/TextField'
import { toast } from '@winterop-com/sokkel-ui/primitives/toast'
import { useState } from 'react'

import { api, asApiError } from '../api/client'
import type { Dataset, GeneratorSettings } from '../api/types'
import { useModel } from '../App'
import { count } from '../lib/format'
import { BOUNDS, generateDefaults, generatedColumns, generateProblem, generateRequest } from '../lib/generate'
import { useResource } from '../lib/resource'
import { configOptions, SelectField } from '../parts'

const PERIOD_TYPES = [
    { value: 'monthly', label: 'Monthly' },
    { value: 'weekly', label: 'Weekly' },
]

/**
 * Makes a dataset with the service's sample-data generator: its settings, defaults from the model
 * (or a generated dataset's `settings`, to make it again), a name and an optional seed.
 */
export function GenerateDialog({
    open,
    onOpenChange,
    settings,
    onGenerated,
}: {
    open: boolean
    onOpenChange: (open: boolean) => void
    settings?: GeneratorSettings
    onGenerated: (dataset: Dataset) => void
}) {
    const model = useModel()
    const configs = useResource(api.configs, [])
    const [form, setForm] = useState(() => generateDefaults(model, settings))
    const [busy, setBusy] = useState(false)
    const [error, setError] = useState<{ title: string; detail?: string }>()
    const set = (patch: Partial<typeof form>) => setForm((was) => ({ ...was, ...patch }))
    const required = model?.required_covariates ?? []
    const config = configs.data?.find((each) => each.id === form.configId)
    const columns = generatedColumns(form, required, config)
    const why = generateProblem(form, columns.length)
    const generate = () => {
        if (why !== undefined || busy) return
        setBusy(true)
        setError(undefined)
        api.generateDataset(generateRequest(form))
            .then((dataset) => {
                toast.add({ type: 'success', title: `${dataset.name} generated`, description: `${count(dataset.rows)} rows, seed ${dataset.generator?.seed ?? '-'}` })
                onGenerated(dataset)
            })
            .catch((failure: unknown) => {
                const refused = asApiError(failure)
                setError({ title: refused.message, detail: refused.detail })
            })
            .finally(() => setBusy(false))
    }
    return (
        <DialogShell
            open={open}
            onOpenChange={onOpenChange}
            title="Generate a dataset"
            description="Synthetic data from the service's sample-data generator, as $generate-sample-data gives it: seasonal climate and cases that follow it."
            confirm={{ label: 'Generate', onConfirm: generate, busy, busyLabel: 'Generating', why }}
        >
            <div className="flex flex-col gap-4">
                <TextField id="generate-name" label="Name" required autoFocus value={form.name} onChange={(name) => set({ name })} placeholder="synthetic-monthly" />
                <div className="grid gap-4 sm:grid-cols-3">
                    <NumberField id="generate-locations" label="Locations" integer min={BOUNDS.locations[0]} max={BOUNDS.locations[1]} value={form.locations} onValueChange={(locations) => set({ locations })} />
                    <NumberField id="generate-periods" label="Periods" integer min={BOUNDS.periods[0]} max={BOUNDS.periods[1]} value={form.periods} onValueChange={(periods) => set({ periods })} />
                    <NumberField
                        id="generate-features"
                        label="Features"
                        hint="feature_N"
                        integer
                        min={BOUNDS.features[0]}
                        max={BOUNDS.features[1]}
                        value={form.features}
                        onValueChange={(features) => set({ features })}
                    />
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                    <SelectField
                        id="generate-period-type"
                        label="Period type"
                        value={form.periodType}
                        options={PERIOD_TYPES}
                        onChange={(periodType) => set({ periodType: periodType === 'weekly' ? 'weekly' : 'monthly' })}
                    />
                    <SelectField
                        id="generate-config"
                        label="Config"
                        help="Adds its additional covariates."
                        value={form.configId}
                        placeholder="None"
                        options={configOptions(configs.data).map((option) => (option.value === '' ? { ...option, label: 'None' } : option))}
                        onChange={(configId) => set({ configId })}
                    />
                </div>
                <NumberField
                    id="generate-seed"
                    label="Seed"
                    hint="optional"
                    help="The same seed and settings make the same data. Left empty, the service draws one and keeps it with the dataset."
                    integer
                    min={0}
                    max={Number.MAX_SAFE_INTEGER}
                    placeholder="Random"
                    value={form.seed}
                    onValueChange={(seed) => set({ seed })}
                />
                <p className="text-sm text-muted-foreground">
                    {form.locations !== null && form.periods !== null ? `${count(form.locations * form.periods)} rows of ${columns.length} columns: ` : `${columns.length} columns: `}
                    {columns.join(', ')}.
                    {required.length > 0 ? ` The model requires ${required.join(', ')}.` : ''}
                </p>
                {error && <Problem title={`The dataset was refused: ${error.title}`} detail={error.detail} />}
            </div>
        </DialogShell>
    )
}
