import { expect, test } from 'bun:test'

import type { Config, Info } from '../api/types'
import { configCovariates, generateDefaults, generatedColumns, generateProblem, generateRequest, MAX_CELLS, sampleDataPath } from './generate'

const info = (period_type: string) => ({ period_type, required_covariates: ['humidity'] }) as unknown as Info
const config = (covariates: unknown) => ({ id: 'c1', name: 'c', data: { additional_continuous_covariates: covariates }, tags: [], created_at: '', updated_at: '' }) as Config

test('defaults are the generator\'s, the period type the model\'s', () => {
    expect(generateDefaults(info('weekly')).periodType).toBe('weekly')
    expect(generateDefaults(info('any')).periodType).toBe('monthly')
    expect(generateDefaults(undefined)).toEqual({ name: '', locations: 5, periods: 50, features: 3, periodType: 'monthly', configId: '', seed: null })
})

test('a generated dataset\'s settings fill the form again, its seed with them', () => {
    const settings = { num_locations: 2, num_periods: 9, num_features: 0, period_type: 'weekly' as const, config_id: 'c1', seed: 42 }
    expect(generateDefaults(undefined, settings)).toEqual({ name: '', locations: 2, periods: 9, features: 0, periodType: 'weekly', configId: 'c1', seed: 42 })
})

test('columns add the features, the required covariates and the config\'s, each once', () => {
    const form = { ...generateDefaults(undefined), features: 2 }
    expect(generatedColumns(form, ['humidity'], config(['humidity', 'wind', 3]))).toEqual([
        'time_period',
        'location',
        'disease_cases',
        'population',
        'rainfall',
        'mean_temperature',
        'feature_0',
        'feature_1',
        'humidity',
        'wind',
    ])
    expect(configCovariates(config('nope'))).toEqual([])
    expect(configCovariates(undefined)).toEqual([])
})

test('the dialog says what is missing, and refuses more cells than the service makes', () => {
    const form = { ...generateDefaults(undefined), name: 'x' }
    expect(generateProblem({ ...form, name: '  ' }, 9)).toBe('Name the dataset')
    expect(generateProblem({ ...form, periods: null }, 9)).toBe('Fill in the numbers')
    expect(generateProblem(form, 9)).toBeUndefined()
    expect(generateProblem({ ...form, locations: 10_000, periods: 10_000 }, 9)).toContain(MAX_CELLS.toLocaleString('en'))
})

test('the request leaves out an empty config and seed, so the service draws one', () => {
    const form = { ...generateDefaults(undefined), name: ' laos ' }
    expect(generateRequest(form)).toEqual({ name: 'laos', num_locations: 5, num_periods: 50, num_features: 3, period_type: 'monthly' })
    expect(generateRequest({ ...form, configId: 'c1', seed: 0 })).toMatchObject({ config_id: 'c1', seed: 0 })
})

test('the settings give the $generate-sample-data query that makes the same data', () => {
    expect(sampleDataPath({ num_locations: 3, num_periods: 14, num_features: 1, period_type: 'weekly', seed: 7 })).toBe(
        '/api/v1/ml/$generate-sample-data?num_locations=3&num_periods=14&num_features=1&period_type=weekly&seed=7',
    )
    expect(sampleDataPath({ num_locations: 3, num_periods: 14, num_features: 1, period_type: 'monthly', config_id: 'c1', seed: 7 })).toContain('&config_id=c1&seed=7')
})
