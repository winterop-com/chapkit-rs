import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { aborted, ApiError, asApiError } from '../api/client'

/** A read: given the signal that stops it once nothing waits for it. */
export type Load<T> = (signal: AbortSignal) => Promise<T>

/** What a read has come to so far. */
export interface ReadState<T> {
    data: T | undefined
    error: ApiError | undefined
    busy: boolean
}

/**
 * Reads `load` now and, with `every`, again that many milliseconds after each read has settled, so
 * two reads never overlap and a slow one is never thrown away for the next. Returns the stop,
 * which aborts the read under way. `report` hears each change.
 */
export function startReading<T>(load: Load<T>, every: number | undefined, report: (patch: Partial<ReadState<T>>) => void): () => void {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    const run = () => {
        report({ busy: true })
        load(controller.signal)
            .then(
                (data) => controller.signal.aborted || report({ data, error: undefined, busy: false }),
                (failure: unknown) => controller.signal.aborted || aborted(failure) || report({ error: asApiError(failure), busy: false }),
            )
            .finally(() => {
                if (!controller.signal.aborted && every !== undefined) timer = setTimeout(run, every)
            })
    }
    run()
    return () => {
        controller.abort()
        if (timer !== undefined) clearTimeout(timer)
    }
}

/** How long to wait before asking about a job again, after `failures` reads in a row failed. */
export const pollDelay = (failures: number) => (failures === 0 ? 400 : Math.min(10_000, 1000 * 2 ** (failures - 1)))

/** What a read holds: its data once it came, whether it is on its way, and how it failed. */
export interface Resource<T> {
    data: T | undefined
    /** Nothing to show yet for this resource, and a read is on its way. */
    loading: boolean
    error: ApiError | undefined
    reload: () => void
}

type Held<T> = ReadState<T> & { owner: object | null }

/**
 * Reads `load` when the deps change, and again `every` milliseconds after each read while given.
 * The data belongs to the deps it was read for: other deps, or no `load`, show none, while a refresh
 * of the same keeps showing the last. A read nothing waits for any more is aborted.
 */
export function useResource<T>(load: Load<T> | null, deps: readonly unknown[], every?: number): Resource<T> {
    // A new object whenever the deps change: the identity the data is kept for.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    const identity = useMemo(() => ({}), deps)
    const [held, setHeld] = useState<Held<T>>({ owner: null, data: undefined, error: undefined, busy: false })
    const [tick, setTick] = useState(0)
    const latest = useRef(load)
    latest.current = load
    const enabled = load !== null
    const reload = useCallback(() => setTick((count) => count + 1), [])

    useEffect(() => {
        const read = latest.current
        const empty: Held<T> = { owner: identity, data: undefined, error: undefined, busy: false }
        if (read === null) {
            setHeld(empty)
            return
        }
        return startReading(read, every, (patch) => setHeld((before) => ({ ...(before.owner === identity ? before : empty), ...patch })))
    }, [identity, tick, every, enabled])

    const own = enabled && held.owner === identity
    return {
        data: own ? held.data : undefined,
        error: own ? held.error : undefined,
        loading: enabled && (!own || (held.busy && held.data === undefined)),
        reload,
    }
}
