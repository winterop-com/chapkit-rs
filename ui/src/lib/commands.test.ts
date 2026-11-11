import { expect, test } from 'bun:test'

import { actionCommands, GO_KEY, PLACE_KEYS, withPlaceKeys } from './commands'

test('place keys follow G, are single letters that need no Alt on a Norwegian keyboard, one place each', () => {
    expect(GO_KEY).toBe('g')
    const keys = Object.values(PLACE_KEYS)
    expect(new Set(keys).size).toBe(keys.length)
    for (const key of keys) expect(key).toMatch(/^[a-z]$/)
})

test('the palette shows G then each place\'s key, and leaves other commands alone', () => {
    const noop = () => {}
    const [group] = withPlaceKeys([
        {
            title: 'Go to',
            commands: [
                { id: 'place-datasets', label: 'Datasets', onSelect: noop },
                { id: 'place-api', label: 'API', onSelect: noop },
            ],
        },
    ])
    // A sequence: G, let go, then D; drawn as G then D, never as one chord.
    expect(group!.commands.map((command) => [command.keys, command.next])).toEqual([
        [['G'], ['D']],
        [undefined, undefined],
    ])
})

test('the actions go to their pages, ask for the key and show the shortcuts', () => {
    const went: string[] = []
    let asked = 0
    let shown = 0
    const [group] = actionCommands({ go: (path) => went.push(path), hasKey: false, askKey: () => asked++, showShortcuts: () => shown++ })
    expect(group!.title).toBe('Actions')
    for (const command of group!.commands) command.onSelect()
    expect(group!.commands.map((command) => command.label)).toEqual([
        'Upload a dataset',
        'Generate a dataset',
        'New evaluation',
        'New forecast',
        'New config',
        'Set API key',
        'Keyboard shortcuts',
    ])
    expect(went).toEqual(['/datasets?upload', '/datasets?generate', '/evaluate', '/forecast', '/configs?new'])
    expect([asked, shown]).toEqual([1, 1])
    const [keyed] = actionCommands({ go: () => {}, hasKey: true, askKey: () => {}, showShortcuts: () => {} })
    expect(keyed!.commands.find((command) => command.id === 'api-key')?.label).toBe('Change API key')
})
