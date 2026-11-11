import { expect, test } from 'bun:test'

import { pollDelay, startReading, type ReadState } from './resource'

const later = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

test('a poll waits for the read under way: no overlap, and a slow read still lands', async () => {
    let running = 0
    let most = 0
    let reads = 0
    const seen: Partial<ReadState<number>>[] = []
    const stop = startReading(
        async () => {
            running += 1
            most = Math.max(most, running)
            reads += 1
            // Slower than the interval: a timer-driven poll would start the next before this one ends.
            await later(30)
            running -= 1
            return reads
        },
        5,
        (patch) => seen.push(patch),
    )
    await later(120)
    stop()
    expect(most).toBe(1)
    expect(reads).toBeGreaterThanOrEqual(2)
    // Every read that finished was reported, in order.
    const landed = seen.flatMap((patch) => (patch.data === undefined ? [] : [patch.data]))
    expect(landed.length).toBeGreaterThanOrEqual(reads - 1)
    expect(landed).toEqual(landed.map((_, at) => at + 1))
})

test('stopping aborts the read under way and reports nothing after it', async () => {
    let signal: AbortSignal | undefined
    const seen: Partial<ReadState<string>>[] = []
    const stop = startReading(
        (given) => {
            signal = given
            return new Promise<string>((resolve, reject) => {
                given.addEventListener('abort', () => reject(new DOMException('stopped', 'AbortError')))
                setTimeout(() => resolve('late'), 50)
            })
        },
        undefined,
        (patch) => seen.push(patch),
    )
    stop()
    await later(80)
    expect(signal?.aborted).toBe(true)
    expect(seen).toEqual([{ busy: true }])
})

test('a failed read is reported as an ApiError and polling goes on', async () => {
    let reads = 0
    const seen: Partial<ReadState<number>>[] = []
    const stop = startReading(
        async () => {
            reads += 1
            if (reads === 1) throw new Error('down')
            return reads
        },
        5,
        (patch) => seen.push(patch),
    )
    await later(40)
    stop()
    expect(seen.find((patch) => patch.error !== undefined)?.error?.message).toBe('Something went wrong')
    expect(seen.some((patch) => patch.data === 2 && patch.error === undefined)).toBe(true)
})

test('a job is asked about again ever more slowly while reads fail', () => {
    expect([0, 1, 2, 3, 4, 5, 9].map(pollDelay)).toEqual([400, 1000, 2000, 4000, 8000, 10_000, 10_000])
})
