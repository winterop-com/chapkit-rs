import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError } from '../api/client'

/** What a read holds: its data once it came, whether it is on its way, and how it failed. */
export interface Resource<T> {
    data: T | undefined
    loading: boolean
    error: ApiError | undefined
    reload: () => void
}

/**
 * Reads `load` when the deps change, and again every `every` milliseconds while given, keeping the
 * last data shown while a new read is on its way.
 */
export function useResource<T>(load: (() => Promise<T>) | null, deps: readonly unknown[], every?: number): Resource<T> {
    const [data, setData] = useState<T>()
    const [loading, setLoading] = useState(load !== null)
    const [error, setError] = useState<ApiError>()
    const [tick, setTick] = useState(0)
    const latest = useRef(load)
    latest.current = load
    const reload = useCallback(() => setTick((count) => count + 1), [])

    useEffect(() => {
        const read = latest.current
        if (read === null) {
            setLoading(false)
            return
        }
        let live = true
        setLoading(true)
        read()
            .then((value) => {
                if (!live) return
                setData(value)
                setError(undefined)
            })
            .catch((failure: unknown) => {
                if (!live) return
                setError(failure instanceof ApiError ? failure : new ApiError(0, 'Something went wrong', String(failure)))
            })
            .finally(() => live && setLoading(false))
        return () => {
            live = false
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [...deps, tick])

    useEffect(() => {
        if (every === undefined) return
        const timer = window.setInterval(reload, every)
        return () => window.clearInterval(timer)
    }, [every, reload])

    return { data, loading: loading && data === undefined, error, reload }
}
