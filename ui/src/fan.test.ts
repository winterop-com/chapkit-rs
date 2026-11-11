import { expect, test } from 'bun:test'

import { fanSpec } from './parts'

test("a fan's data view names every field it shows and writes figures as scores", () => {
    const spec = fanSpec([{ period: '2024-02', median: 12.345, q10: 1, q25: 2, q75: 3, q90: 4 }, { period: '2024-01', observed: 1234.5 }])
    const fields = [spec.x, spec.y, spec.points, ...spec.bands.flatMap((band) => [band.low, band.high])]
    expect(fields.map((field) => spec.fieldNames[field])).toEqual(['Period', 'Median', 'Observed', '10th', '90th', '25th', '75th'])
    expect(spec.format(12.345)).toBe('12.3')
    expect(spec.rows.map((row) => row.period)).toEqual(['2024-01', '2024-02'])
    expect(spec.marks).toEqual([{ x: '2024-02', label: 'Forecast' }])
})
