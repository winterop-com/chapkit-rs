import type { PaletteCommand, PaletteGroup } from '@winterop-com/sokkel-ui/CommandPalette'
import { bindingKeys } from '@winterop-com/sokkel-ui/lib/keys'
import { Gauge, Keyboard, KeyRound, Plus, TrendingUp, Upload } from 'lucide-react'

/** The first key of every way to a place: G, then the place's key. */
export const GO_KEY = 'g'

/**
 * The key after G that goes to each place from anywhere outside a field, the place's first letter
 * where it is free. None needs Alt on a Norwegian keyboard.
 */
export const PLACE_KEYS: Readonly<Record<string, string>> = {
    overview: 'o',
    datasets: 'd',
    evaluate: 'e',
    forecast: 'f',
    configs: 'c',
    jobs: 'j',
    artifacts: 'a',
    system: 's',
}

/** The keys that go to a place, G then the place's key; undefined for a place without one. */
export function placeBinding(place: string): { key: string; next: string } | undefined {
    const next = PLACE_KEYS[place]
    return next === undefined ? undefined : { key: GO_KEY, next }
}

/** The palette's places with the keys that go to each, G then the place's key, drawn at the row's end. */
export function withPlaceKeys(groups: readonly PaletteGroup[]): PaletteGroup[] {
    return groups.map((group) => ({
        ...group,
        commands: group.commands.map((command) => {
            const binding = placeBinding(command.id.replace(/^place-/, ''))
            return binding === undefined ? command : { ...command, ...bindingKeys(binding) }
        }),
    }))
}

/** The title of the palette's group of actions; a page's own command of the same id takes the app's place. */
export const ACTIONS = 'Actions'

/** The ids of the actions a page answers itself while it is open. */
export const UPLOAD_DATASET = 'upload-dataset'
export const NEW_CONFIG = 'new-config'

/** What the actions do, the app's to give. */
export interface ActionHandlers {
    go: (path: string) => void
    /** Whether an API key is set, so the command says set or change. */
    hasKey: boolean
    askKey: () => void
    showShortcuts: () => void
}

/**
 * The palette's actions: upload a dataset, new evaluation, forecast and config, the API key and the
 * keyboard shortcuts. Upload and new config go to their page with the dialog asked for (`?upload`,
 * `?new`); on that page the page's own command, of the same id, opens it in place.
 */
export function actionCommands({ go, hasKey, askKey, showShortcuts }: ActionHandlers): PaletteGroup[] {
    const commands: PaletteCommand[] = [
        { id: UPLOAD_DATASET, label: 'Upload a dataset', icon: Upload, keywords: ['csv', 'json', 'file'], onSelect: () => go('/datasets?upload') },
        { id: 'new-evaluation', label: 'New evaluation', icon: Gauge, keywords: ['backtest', 'run'], onSelect: () => go('/evaluate') },
        { id: 'new-forecast', label: 'New forecast', icon: TrendingUp, keywords: ['run'], onSelect: () => go('/forecast') },
        { id: NEW_CONFIG, label: 'New config', icon: Plus, keywords: ['options'], onSelect: () => go('/configs?new') },
        { id: 'api-key', label: hasKey ? 'Change API key' : 'Set API key', icon: KeyRound, keywords: ['x-api-key', 'token'], onSelect: askKey },
        { id: 'shortcuts', label: 'Keyboard shortcuts', icon: Keyboard, keys: ['?'], keywords: ['keys', 'help'], onSelect: showShortcuts },
    ]
    return [{ title: ACTIONS, commands, rank: 10, verb: 'Run' }]
}
