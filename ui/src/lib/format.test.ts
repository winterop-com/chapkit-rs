import { expect, test } from 'bun:test'
import { score } from './format'

test('score keeps significant zeros and drops padding', () => {
    expect(score(140)).toBe('140')
    expect(score(100)).toBe('100')
    expect(score(90.6)).toBe('90.6')
    expect(score(7.9)).toBe('7.9')
    expect(score(2.5)).toBe('2.5')
    expect(score(0.5)).toBe('0.5')
    expect(score(14)).toBe('14')
    expect(score(1234.4)).toBe('1,234')
    expect(score(0.397, true)).toBe('39.7%')
    expect(score(null)).toBe('-')
})
