import { useEffect, useState } from 'react'

/** The hash route as its parts: `#/datasets/abc` is `['datasets', 'abc']`. */
function read(): string[] {
    return window.location.hash
        .replace(/^#\/?/, '')
        .split('?')[0]!
        .split('/')
        .filter((part) => part !== '')
        .map(decodeURIComponent)
}

/** The hash route, followed as it changes. */
export function useRoute(): string[] {
    const [route, setRoute] = useState(read)
    useEffect(() => {
        const change = () => setRoute(read())
        window.addEventListener('hashchange', change)
        return () => window.removeEventListener('hashchange', change)
    }, [])
    return route
}

/** Goes to a route such as `/datasets`. */
export function go(path: string): void {
    window.location.hash = path
}

/** A parameter of the hash route's query: `#/evaluate?dataset=abc` has `dataset`. */
export function hashParam(name: string): string | undefined {
    const query = window.location.hash.split('?')[1]
    return query === undefined ? undefined : (new URLSearchParams(query).get(name) ?? undefined)
}
