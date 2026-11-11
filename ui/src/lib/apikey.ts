import { useSyncExternalStore } from 'react'

/**
 * The service's API key (`X-API-Key`), when it asks for one. Kept for the tab in session storage,
 * never in a URL; a page whose storage is blocked keeps it in memory.
 */
const STORAGE = 'chapkit-api-key'

interface KeyState {
    /** The key sent with every request; undefined while none is given. */
    key: string | undefined
    /** A request was refused for want of a key, or for a wrong one. */
    wanted: boolean
    /** Counts the keys given, so reads made with an older key can be made again. */
    version: number
}

function stored(): string | undefined {
    try {
        return globalThis.sessionStorage?.getItem(STORAGE) ?? undefined
    } catch {
        return undefined
    }
}

let state: KeyState = { key: stored(), wanted: false, version: 0 }
const listeners = new Set<() => void>()

function change(next: Partial<KeyState>): void {
    state = { ...state, ...next }
    for (const listener of listeners) listener()
}

/** The key to send, if any. */
export const apiKey = (): string | undefined => state.key

/** Whether a request was refused for want of a key since one was last given. */
export const keyWanted = (): boolean => state.wanted

/** Gives a key (or takes it away with undefined), and asks the page to read again with it. */
export function setApiKey(key: string | undefined): void {
    const given = key?.trim() || undefined
    try {
        if (given === undefined) globalThis.sessionStorage?.removeItem(STORAGE)
        else globalThis.sessionStorage?.setItem(STORAGE, given)
    } catch {
        // Storage blocked: the key lives as long as the page.
    }
    change({ key: given, wanted: false, version: state.version + 1 })
}

/** The generation of the key in use, which a request keeps to say which key it was sent with. */
export const keyVersion = (): number => state.version

/**
 * Says a request sent with the key of generation `version` was refused with 401: the key is
 * missing or wrong. A refusal of an older key, answered after a new one was given, is ignored.
 */
export function keyRefused(version: number): void {
    if (version === state.version && !state.wanted) change({ wanted: true })
}

/** Stops asking for a key until a request is refused again. */
export const dismissKeyRequest = (): void => change({ wanted: false })

const subscribe = (listener: () => void) => {
    listeners.add(listener)
    return () => listeners.delete(listener)
}

/** The key state, followed as it changes. */
export const useApiKey = (): KeyState => useSyncExternalStore(subscribe, () => state)
