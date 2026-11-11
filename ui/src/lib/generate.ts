import type { Config, GenerateRequest, GeneratorSettings, Info } from '../api/types'

/** The most cells (locations x periods x columns) the service generates. */
export const MAX_CELLS = 20_000_000

/** The bounds the service holds each setting to. */
export const BOUNDS = { locations: [1, 10_000], periods: [1, 10_000], features: [0, 100] } as const

/** The columns every generated dataset has, before the features and the covariates. */
const BASE_COLUMNS = ['time_period', 'location', 'disease_cases', 'population', 'rainfall', 'mean_temperature']

/** The generator's settings as the dialog holds them: a number not typed yet is null. */
export interface GenerateForm {
    name: string
    locations: number | null
    periods: number | null
    features: number | null
    periodType: 'monthly' | 'weekly'
    /** A config whose covariates are added; '' for none. */
    configId: string
    /** null: a random seed. */
    seed: number | null
}

/** The generator's own defaults, the period type the model's. */
export function generateDefaults(info: Info | undefined, settings?: GeneratorSettings): GenerateForm {
    if (settings !== undefined)
        return {
            name: '',
            locations: settings.num_locations,
            periods: settings.num_periods,
            features: settings.num_features,
            periodType: settings.period_type,
            configId: settings.config_id ?? '',
            seed: settings.seed,
        }
    return { name: '', locations: 5, periods: 50, features: 3, periodType: info?.period_type === 'weekly' ? 'weekly' : 'monthly', configId: '', seed: null }
}

/** A config's `additional_continuous_covariates`. */
export function configCovariates(config: Config | undefined): string[] {
    const covariates = config?.data.additional_continuous_covariates
    return Array.isArray(covariates) ? covariates.filter((each): each is string => typeof each === 'string') : []
}

/** The columns the dataset will have: the generator's, the features, the model's required covariates, the config's. */
export function generatedColumns(form: GenerateForm, required: readonly string[], config: Config | undefined): string[] {
    const columns = [...BASE_COLUMNS, ...Array.from({ length: form.features ?? 0 }, (_, index) => `feature_${index}`)]
    for (const column of [...required, ...configCovariates(config)]) if (!columns.includes(column)) columns.push(column)
    return columns
}

/** Why the dataset cannot be generated yet, or undefined when it can. */
export function generateProblem(form: GenerateForm, columns: number): string | undefined {
    if (form.name.trim() === '') return 'Name the dataset'
    if (form.locations === null || form.periods === null || form.features === null) return 'Fill in the numbers'
    if (form.locations * form.periods * columns > MAX_CELLS) return `More than ${MAX_CELLS.toLocaleString('en')} cells; ask for fewer locations or periods`
    return undefined
}

/** The request the form makes; call it only when `generateProblem` finds nothing. */
export function generateRequest(form: GenerateForm): GenerateRequest {
    return {
        name: form.name.trim(),
        num_locations: form.locations ?? 5,
        num_periods: form.periods ?? 50,
        num_features: form.features ?? 3,
        period_type: form.periodType,
        ...(form.configId === '' ? {} : { config_id: form.configId }),
        ...(form.seed === null ? {} : { seed: form.seed }),
    }
}

/**
 * The `$generate-sample-data` query that gives the same data as a generated dataset: the same
 * settings make it again.
 */
export function sampleDataPath(settings: GeneratorSettings): string {
    const params = new URLSearchParams({
        num_locations: String(settings.num_locations),
        num_periods: String(settings.num_periods),
        num_features: String(settings.num_features),
        period_type: settings.period_type,
    })
    if (settings.config_id !== undefined) params.set('config_id', settings.config_id)
    params.set('seed', String(settings.seed))
    return `/api/v1/ml/$generate-sample-data?${params.toString()}`
}
