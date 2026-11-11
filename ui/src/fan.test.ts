import { expect, test } from 'bun:test'
import { dataColumns } from '@winterop-com/sokkel-ui-charts/data-columns'

import { datasetSpec } from './pages/DatasetPage'
import { fanSpec } from './parts'

test("a fan's data view names every field it shows and writes figures as scores", () => {
    const spec = fanSpec([
        { period: '2024-02', location: 'Bokeo', ahead: 1, median: 12.345, q10: 1, q25: 2, q75: 3, q90: 4 },
        { period: '2024-01', location: 'Bokeo', observed: 1234.5 },
    ])
    const fields = [spec.x, spec.y, spec.points, ...spec.bands.flatMap((band) => [band.low, band.high])]
    expect(fields.map((field) => spec.fieldNames[field])).toEqual(['Period', 'Median', 'Observed', '10th', '90th', '25th', '75th'])
    expect(spec.dataFields.map((field) => spec.fieldNames[field])).toEqual(['Location', 'Periods ahead'])
    expect(spec.format(12.345)).toBe('12.3')
    expect(spec.rows.map((row) => [row.period, row.location, row.ahead])).toEqual([
        ['2024-01', 'Bokeo', undefined],
        ['2024-02', 'Bokeo', 1],
    ])
    expect(spec.marks).toEqual([{ x: '2024-02', label: 'Forecast' }])
})

test("the evaluation and forecast fans' data views read Period, the quantiles in order, Observed, Location and Periods ahead", () => {
    const evaluation = fanSpec([
        { period: '2020-04', location: 'Bokeo', observed: 3 },
        { period: '2020-05', location: 'Bokeo', ahead: 1, observed: 4, median: 5, q10: 1, q25: 2, q75: 6, q90: 8 },
    ])
    const forecast = fanSpec([{ period: '2020-05', location: 'Bokeo', ahead: 1, median: 5, q10: 1, q25: 2, q75: 6, q90: 8 }])
    for (const spec of [evaluation, forecast]) {
        expect(dataColumns('fan', spec).map((column) => [column.field, column.header])).toEqual([
            ['period', 'Period'],
            ['q10', '10th'],
            ['q25', '25th'],
            ['median', 'Median'],
            ['q75', '75th'],
            ['q90', '90th'],
            ['observed', 'Observed'],
            ['location', 'Location'],
            ['ahead', 'Periods ahead'],
        ])
    }
})

test("the dataset chart's data view reads Period, Location and the target", () => {
    const spec = datasetSpec([{ _key: 'a', time_period: '2020-05', location: 'Bokeo', disease_cases: 7 }], 'disease_cases', ['Bokeo'])
    expect(dataColumns('line', spec).map((column) => [column.field, column.header])).toEqual([
        ['period', 'Period'],
        ['location', 'Location'],
        ['value', 'disease_cases'],
    ])
})
