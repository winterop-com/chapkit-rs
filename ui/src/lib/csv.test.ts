import { expect, test } from 'bun:test'

import { inert, toCsv } from './csv'

test('text a spreadsheet would run as a formula is made inert', () => {
    for (const formula of ['=1+1', '+1', '-1', '@SUM(A1)', '\tx', '\rx', '=HYPERLINK("x")']) expect(inert(formula)).toBe(`'${formula}`)
    expect(inert('Bokeo')).toBe('Bokeo')
    expect(inert('a=b')).toBe('a=b')
    expect(inert(-1.5)).toBe('-1.5')
    expect(inert(null)).toBe('')
})

test('a forecast export neutralises locations and keeps numbers', () => {
    const rows = [
        { location: '=1+1', time_period: '2024-01', median: -2 },
        { location: 'a,"b"', time_period: '2024-02', median: 3.5 },
    ]
    expect(toCsv(rows, ['location', 'time_period', 'median'])).toBe(`location,time_period,median\n'=1+1,2024-01,-2\n"a,""b""",2024-02,3.5\n`)
})
